"""Adam 精调：Huber 重投影 + 关节限幅 + 时序正则（NumPy + 解析 FK 反传）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.geometry.body_fk import coco_joints_from_fk, fk_forward
from core.geometry.calib_model import StereoCalibration
from core.geometry.rotations import rodrigues
from core.geometry.soma_skeleton import SomaSkeleton


@dataclass
class BodyLossWeights:
    w_reproj_l: float = 1.0
    w_reproj_r: float = 1.0
    w_pose_prior: float = 0.001
    w_limit: float = 0.01
    w_temporal: float = 0.5
    w_twist: float = 0.1
    huber_delta: float = 1.0


@dataclass
class BodyOptimizerConfig:
    max_iter_first_frame: int = 30
    max_iter_tracking: int = 10
    learning_rate: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    early_stop_patience: int = 3
    early_stop_tolerance: float = 0.999
    grad_clip_norm: float = 1.0
    limit_stiffness: float = 10.0


def _huber_scalar(d: float, delta: float) -> tuple[float, float]:
    ad = abs(d)
    if ad < delta:
        return 0.5 * d * d, d
    return delta * (ad - 0.5 * delta), delta * (1.0 if d > 0 else -1.0)


def _project(
    pts: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cam = (R @ pts.T).T + t.reshape(1, 3)
    z = np.maximum(cam[:, 2], 1e-6)
    u = K[0, 0] * cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1), cam, z


def _proj_grad(
    X: np.ndarray,
    K: np.ndarray,
    R: np.ndarray | None,
    t: np.ndarray | None,
    axis: int,
) -> np.ndarray:
    if R is None:
        z = max(float(X[2]), 1e-6)
        if axis == 0:
            return np.array([K[0, 0] / z, 0.0, -K[0, 0] * X[0] / (z * z)], dtype=np.float64)
        return np.array([0.0, K[1, 1] / z, -K[1, 1] * X[1] / (z * z)], dtype=np.float64)
    assert t is not None
    Xr = R @ X + t
    z = max(float(Xr[2]), 1e-6)
    if axis == 0:
        d_cam = np.array([K[0, 0] / z, 0.0, -K[0, 0] * Xr[0] / (z * z)], dtype=np.float64)
    else:
        d_cam = np.array([0.0, K[1, 1] / z, -K[1, 1] * Xr[1] / (z * z)], dtype=np.float64)
    return R.T @ d_cam


def _soft_limit_loss(pose: np.ndarray, skeleton: SomaSkeleton, stiffness: float) -> float:
    loss = 0.0
    for j in range(77):
        joint_idx = j + 1
        lim = skeleton.joint_limits[joint_idx]
        for axis in range(3):
            val = float(pose[j, axis])
            x_min, x_max = float(lim[axis * 2]), float(lim[axis * 2 + 1])
            loss += float(np.log1p(np.exp(stiffness * (val - x_max))))
            loss += float(np.log1p(np.exp(stiffness * (x_min - val))))
    return loss


def _axis_angle_grad_from_rotmat(local_rot: np.ndarray, dL_dR: np.ndarray) -> np.ndarray:
    eps = 1e-5
    grad = np.zeros(3, dtype=np.float64)
    base = rodrigues(np.zeros(3))
    for a in range(3):
        aa = np.zeros(3, dtype=np.float64)
        aa[a] = eps
        plus = rodrigues(aa)
        grad[a] = float(np.sum(dL_dR * (plus - base)) / eps)
    _ = local_rot
    return grad


def fk_backward(
    joint_grad: np.ndarray,
    pose: np.ndarray,
    skeleton: SomaSkeleton,
    world_rots: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = skeleton.num_joints
    parent = skeleton.parent
    rest = skeleton.rest_local.astype(np.float64)
    grad_pose = np.zeros((77, 3), dtype=np.float64)
    pos_grad = joint_grad.astype(np.float64).copy()
    rot_grad = np.zeros((n, 3, 3), dtype=np.float64)

    for j in reversed(range(1, n)):
        p = int(parent[j])
        pos_grad[p] += pos_grad[j]
        rot_grad[p] += np.outer(pos_grad[j], rest[j])
        local_rot = rodrigues(pose[j - 1])
        rot_grad[p] += rot_grad[j] @ local_rot.T
        dL_dlocal = world_rots[p].T @ rot_grad[j]
        grad_pose[j - 1] += _axis_angle_grad_from_rotmat(local_rot, dL_dlocal)

    return grad_pose.astype(np.float32), pos_grad[0].astype(np.float32)


def _compute_loss_and_grad(
    pose: np.ndarray,
    trans: np.ndarray,
    pose_prev: np.ndarray | None,
    pose_prev2: np.ndarray | None,
    frame_id: int,
    obs_l: np.ndarray,
    obs_r: np.ndarray,
    conf_l: np.ndarray,
    conf_r: np.ndarray,
    skeleton: SomaSkeleton,
    calib: StereoCalibration,
    weights: BodyLossWeights,
    opt_cfg: BodyOptimizerConfig,
    min_conf: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    joints, world_rots = fk_forward(pose, trans, skeleton)
    coco_pts = coco_joints_from_fk(joints, skeleton)
    K1, K2 = calib.K1.astype(np.float64), calib.K2.astype(np.float64)
    R2 = calib.R.astype(np.float64)
    t2 = calib.T.reshape(3).astype(np.float64)

    proj_l, _, _ = _project(coco_pts.astype(np.float64), K1, np.eye(3), np.zeros(3))
    proj_r, _, _ = _project(coco_pts.astype(np.float64), K2, R2, t2)

    joint_grad = np.zeros_like(joints, dtype=np.float64)
    loss = 0.0
    delta = weights.huber_delta
    for i in range(17):
        c = min(float(conf_l[i]), float(conf_r[i]))
        if c < min_conf:
            continue
        for cam_w, proj, obs, conf, R, t in (
            (weights.w_reproj_l, proj_l, obs_l, conf_l, None, None),
            (weights.w_reproj_r, proj_r, obs_r, conf_r, R2, t2),
        ):
            diff = proj[i] - obs[i].astype(np.float64)
            X = coco_pts[i].astype(np.float64)
            for axis in range(2):
                h, dloss_dd = _huber_scalar(float(diff[axis]), delta)
                loss += cam_w * float(conf[i]) * h
                dproj = _proj_grad(X, K1 if R is None else K2, R, t, axis)
                soma_idx = int(skeleton.coco_to_soma[i])
                joint_grad[soma_idx] += cam_w * float(conf[i]) * dloss_dd * dproj

    loss += weights.w_pose_prior * float(np.sum(pose * pose))
    grad_pose = (2.0 * weights.w_pose_prior * pose.astype(np.float64)).astype(np.float32)

    if frame_id >= 2 and pose_prev is not None and pose_prev2 is not None:
        delta_t = pose.astype(np.float64) - pose_prev.astype(np.float64)
        delta_t1 = pose_prev.astype(np.float64) - pose_prev2.astype(np.float64)
        acc = delta_t - delta_t1
        loss += weights.w_temporal * float(np.sum(acc * acc))
        grad_pose += (2.0 * weights.w_temporal * acc).astype(np.float32)
    elif frame_id == 1 and pose_prev is not None:
        vel = pose.astype(np.float64) - pose_prev.astype(np.float64)
        loss += weights.w_temporal * float(np.sum(vel * vel))
        grad_pose += (2.0 * weights.w_temporal * vel).astype(np.float32)

    if frame_id > 0 and pose_prev is not None:
        for tj in skeleton.twist_joints:
            pi = skeleton.pose_joint_index(tj)
            dt = float(pose[pi, 0] - pose_prev[pi, 0])
            loss += weights.w_twist * dt * dt
            grad_pose[pi, 0] += np.float32(2.0 * weights.w_twist * dt)

    loss += weights.w_limit * _soft_limit_loss(pose, skeleton, opt_cfg.limit_stiffness)

    fk_g_pose, fk_g_trans = fk_backward(joint_grad, pose, skeleton, world_rots)
    grad_pose = grad_pose.astype(np.float64) + fk_g_pose.astype(np.float64)
    return loss, grad_pose.astype(np.float32), fk_g_trans.astype(np.float32)


def optimize_pose(
    pose_init: np.ndarray,
    trans_init: np.ndarray,
    pose_prev: np.ndarray | None,
    pose_prev2: np.ndarray | None,
    frame_id: int,
    obs_l: np.ndarray,
    obs_r: np.ndarray,
    conf_l: np.ndarray,
    conf_r: np.ndarray,
    skeleton: SomaSkeleton,
    calib: StereoCalibration,
    *,
    weights: BodyLossWeights | None = None,
    opt_cfg: BodyOptimizerConfig | None = None,
    min_conf: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, float]:
    weights = weights or BodyLossWeights()
    opt_cfg = opt_cfg or BodyOptimizerConfig()
    pose = np.asarray(pose_init, dtype=np.float32).copy()
    trans = np.asarray(trans_init, dtype=np.float32).copy()
    max_iter = (
        opt_cfg.max_iter_first_frame if frame_id == 0 else opt_cfg.max_iter_tracking
    )

    m_p = np.zeros_like(pose)
    v_p = np.zeros_like(pose)
    m_t = np.zeros(3, dtype=np.float32)
    v_t = np.zeros(3, dtype=np.float32)
    beta1, beta2 = opt_cfg.beta1, opt_cfg.beta2
    lr = opt_cfg.learning_rate
    prev_loss = float("inf")
    stall = 0
    final_loss = prev_loss

    for t in range(1, max_iter + 1):
        loss, g_p, g_t = _compute_loss_and_grad(
            pose,
            trans,
            pose_prev,
            pose_prev2,
            frame_id,
            obs_l,
            obs_r,
            conf_l,
            conf_r,
            skeleton,
            calib,
            weights,
            opt_cfg,
            min_conf,
        )
        final_loss = loss
        g_norm = float(np.linalg.norm(g_p) + np.linalg.norm(g_t))
        if g_norm > opt_cfg.grad_clip_norm:
            scale = opt_cfg.grad_clip_norm / max(g_norm, 1e-8)
            g_p = (g_p * scale).astype(np.float32)
            g_t = (g_t * scale).astype(np.float32)

        m_p = beta1 * m_p + (1.0 - beta1) * g_p
        v_p = beta2 * v_p + (1.0 - beta2) * (g_p * g_p)
        m_t = beta1 * m_t + (1.0 - beta2) * g_t
        v_t = beta2 * v_t + (1.0 - beta2) * (g_t * g_t)
        bc1 = 1.0 - beta1**t
        bc2 = 1.0 - beta2**t
        pose -= (lr * (m_p / bc1) / (np.sqrt(v_p / bc2) + 1e-8)).astype(np.float32)
        trans -= (lr * (m_t / bc1) / (np.sqrt(v_t / bc2) + 1e-8)).astype(np.float32)

        if loss > prev_loss * opt_cfg.early_stop_tolerance:
            stall += 1
            if stall >= opt_cfg.early_stop_patience:
                break
        else:
            stall = 0
        prev_loss = loss

    return pose, trans, float(final_loss)
