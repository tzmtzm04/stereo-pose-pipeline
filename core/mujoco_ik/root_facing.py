"""观测肩–髋定 root 朝向；IK 冻结 freejoint 旋转（方案 C）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

SiteDict = dict[str, np.ndarray | None]

RootFacingMode = Literal["yaw_correct", "torso_frame", "yaw_only"]


@dataclass(frozen=True)
class RootFacingConfig:
    enabled: bool = True
    mode: RootFacingMode = "yaw_correct"
    up_blend: float = 0.15
    yaw_unwrap: bool = True
    lock_translation: bool = False
    hard_lock_rotation: bool = False


@dataclass(frozen=True)
class RootFacingLock:
    quat: np.ndarray
    pos: np.ndarray | None = None
    hard_lock_rotation: bool = False


def _site(targets: SiteDict, name: str) -> np.ndarray | None:
    p = targets.get(name)
    if p is None:
        return None
    arr = np.asarray(p, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def estimate_torso_frame(
    targets: SiteDict,
    *,
    up_blend: float = 0.15,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    pelvis = _site(targets, "pelvis")
    if pelvis is None:
        lh = _site(targets, "left_hip")
        rh = _site(targets, "right_hip")
        if lh is not None and rh is not None:
            pelvis = 0.5 * (lh + rh)
        elif lh is not None:
            pelvis = lh
        elif rh is not None:
            pelvis = rh
        else:
            return None

    ls = _site(targets, "left_shoulder")
    rs = _site(targets, "right_shoulder")
    if ls is None or rs is None:
        return None

    top = 0.5 * (ls + rs)
    lateral = ls - rs

    up = top - pelvis
    un = float(np.linalg.norm(up))
    if un < 1e-6:
        return None
    up = up / un
    blend = float(up_blend)
    if blend > 0.0:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up = up + blend * world_up
        up /= np.linalg.norm(up) + 1e-9

    ln = float(np.linalg.norm(lateral))
    if ln < 1e-6:
        return None
    lateral = lateral / ln

    forward = np.cross(up, lateral)
    fn = float(np.linalg.norm(forward))
    if fn < 1e-6:
        return None
    forward = forward / fn

    right = np.cross(forward, up)
    rn = float(np.linalg.norm(right))
    if rn < 1e-6:
        return None
    right = right / rn

    up = np.cross(right, forward)
    up /= np.linalg.norm(up) + 1e-9
    return right, up, forward


def estimate_torso_yaw(
    targets: SiteDict,
    *,
    up_blend: float = 0.15,
) -> float | None:
    frame = estimate_torso_frame(targets, up_blend=up_blend)
    if frame is None:
        return None
    _right, _up, forward = frame
    return float(np.arctan2(forward[0], forward[2]))


def unwrap_yaw(yaw: float, prev: float | None) -> float:
    if prev is None:
        return float(yaw)
    dy = (float(yaw) - float(prev) + np.pi) % (2.0 * np.pi) - np.pi
    return float(prev) + dy


def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    m = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def torso_frame_to_quat_wxyz(
    frame: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    right, up, forward = frame
    R = np.column_stack(
        [
            np.asarray(right, dtype=np.float64),
            np.asarray(up, dtype=np.float64),
            np.asarray(forward, dtype=np.float64),
        ]
    )
    return _mat_to_quat_wxyz(R)


def yaw_to_quat_wxyz(yaw: float) -> np.ndarray:
    half = 0.5 * float(yaw)
    q = np.array([np.cos(half), 0.0, np.sin(half), 0.0], dtype=np.float64)
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return q / n


def quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = np.asarray(a, dtype=np.float64).reshape(4)
    bw, bx, by, bz = np.asarray(b, dtype=np.float64).reshape(4)
    q = np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )
    return q / (np.linalg.norm(q) + 1e-12)


def quat_rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    vq = np.array([0.0, *np.asarray(v, dtype=np.float64).reshape(3)], dtype=np.float64)
    qc = np.array(
        [q[0], -q[1], -q[2], -q[3]],
        dtype=np.float64,
    )
    return quat_mul_wxyz(quat_mul_wxyz(q, vq), qc)[1:4]


def quat_forward_yaw(q: np.ndarray) -> float:
    fwd = quat_rotate_vec(q, np.array([0.0, 0.0, 1.0], dtype=np.float64))
    return float(np.arctan2(fwd[0], fwd[2]))


def align_quat_hemisphere(quat: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    if prev is None:
        return q
    p = np.asarray(prev, dtype=np.float64).reshape(4)
    if float(np.dot(q, p)) < 0.0:
        return -q
    return q


def correct_root_yaw(
    quat: np.ndarray,
    yaw_target: float,
) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    q = q / (np.linalg.norm(q) + 1e-12)
    yaw_cur = quat_forward_yaw(q)
    delta = float(yaw_target) - yaw_cur
    delta = (delta + np.pi) % (2.0 * np.pi) - np.pi
    return quat_mul_wxyz(yaw_to_quat_wxyz(delta), q)


def apply_root_lock(qpos: np.ndarray, lock: RootFacingLock) -> np.ndarray:
    q = np.asarray(qpos, dtype=np.float64).copy()
    if lock.pos is not None:
        q[0:3] = np.asarray(lock.pos, dtype=np.float64).reshape(3)
    q[3:7] = np.asarray(lock.quat, dtype=np.float64).reshape(4)
    return q


def freejoint_v_ids(model, mujoco_mod) -> np.ndarray:
    ids: list[int] = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(mujoco_mod.mjtJoint.mjJNT_FREE):
            continue
        vadr = int(model.jnt_dofadr[j])
        ids.extend(range(vadr, vadr + 6))
    return np.asarray(ids, dtype=int)


def freejoint_rotation_v_ids(model, mujoco_mod) -> np.ndarray:
    ids: list[int] = []
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(mujoco_mod.mjtJoint.mjJNT_FREE):
            continue
        vadr = int(model.jnt_dofadr[j])
        ids.extend(range(vadr + 3, vadr + 6))
    return np.asarray(ids, dtype=int)


def model_root_on_pelvis(model, mujoco_mod) -> bool:
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != int(mujoco_mod.mjtJoint.mjJNT_FREE):
            continue
        bid = int(model.jnt_bodyid[j])
        name = mujoco_mod.mj_id2name(model, mujoco_mod.mjtObj.mjOBJ_BODY, bid) or ""
        return name.lower() == "pelvis"
    return False


class RootFacingTracker:
    """逐帧从 IK targets 生成 RootFacingLock。"""

    def __init__(self, cfg: RootFacingConfig | None = None) -> None:
        self.cfg = cfg or RootFacingConfig()
        self._prev_yaw: float | None = None
        self._prev_quat: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_yaw = None
        self._prev_quat = None

    def lock_from_targets(
        self,
        targets: SiteDict,
        qpos_fallback: np.ndarray | None = None,
    ) -> RootFacingLock | None:
        if not self.cfg.enabled:
            return None

        yaw_raw = estimate_torso_yaw(targets, up_blend=self.cfg.up_blend)
        if yaw_raw is None:
            return None
        yaw = (
            unwrap_yaw(yaw_raw, self._prev_yaw)
            if self.cfg.yaw_unwrap
            else float(yaw_raw)
        )
        self._prev_yaw = yaw

        if self.cfg.mode == "yaw_correct" and qpos_fallback is not None:
            q_prev = np.asarray(qpos_fallback[3:7], dtype=np.float64)
            quat = correct_root_yaw(q_prev, yaw)
        elif self.cfg.mode == "torso_frame":
            frame = estimate_torso_frame(targets, up_blend=self.cfg.up_blend)
            if frame is None:
                return None
            quat = torso_frame_to_quat_wxyz(frame)
        else:
            quat = yaw_to_quat_wxyz(yaw)

        quat = align_quat_hemisphere(quat, self._prev_quat)
        self._prev_quat = quat.copy()

        pos: np.ndarray | None = None
        if self.cfg.lock_translation:
            pelvis = _site(targets, "pelvis")
            if pelvis is not None:
                pos = pelvis.copy()

        return RootFacingLock(
            quat=quat,
            pos=pos,
            hard_lock_rotation=bool(self.cfg.hard_lock_rotation),
        )

    def correct_qpos_root(
        self,
        qpos: np.ndarray,
        targets: SiteDict,
    ) -> np.ndarray | None:
        """IK 后仅修正 root yaw，保留 pitch/roll（yaw_correct 模式）。"""
        if not self.cfg.enabled or self.cfg.mode != "yaw_correct":
            return None
        yaw_raw = estimate_torso_yaw(targets, up_blend=self.cfg.up_blend)
        if yaw_raw is None:
            return None
        yaw = (
            unwrap_yaw(yaw_raw, self._prev_yaw)
            if self.cfg.yaw_unwrap
            else float(yaw_raw)
        )
        self._prev_yaw = yaw
        q = np.asarray(qpos, dtype=np.float64).copy()
        q[3:7] = correct_root_yaw(q[3:7], yaw)
        self._prev_quat = q[3:7].copy()
        return q
