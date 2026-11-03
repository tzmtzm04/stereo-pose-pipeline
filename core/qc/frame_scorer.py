"""逐帧 QC 打分（规则层，无神经网络）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from core.geometry.calib_model import StereoCalibration
from core.qc.h36m_joints import coco_kpts3d_to_h36m, h36m_coord_for_export

# COCO 骨段（用于长度一致性）
_BONE_PAIRS = (
    (5, 6),
    (11, 12),
    (5, 11),
    (6, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
)


@dataclass
class QcConfig:
    min_conf: float = 0.25
    min_valid_joints: int = 10
    min_z_m: float = 0.15
    max_z_m: float = 6.0
    max_joint_z_m: float = 8.0
    require_tri_ok: bool = True
    pass_score_threshold: float = 0.55
    w_conf: float = 0.30
    w_valid: float = 0.20
    w_reproj: float = 0.25
    w_bone: float = 0.15
    w_velocity: float = 0.10
    reproj_px_scale: float = 40.0
    bone_ratio_tol: float = 0.45
    max_bone_vel_m_s: float = 8.0
    fps: float = 60.0


@dataclass
class QcFrameResult:
    qc_score: float
    qc_pass: bool
    qc_flags: list[str] = field(default_factory=list)
    mean_conf: float = 0.0
    valid_joint_count: int = 0
    reproj_px: float = 0.0
    bone_outlier_ratio: float = 0.0
    max_joint_speed_m_s: float = 0.0
    h36m_kpts: np.ndarray | None = None
    h36m_valid: np.ndarray | None = None
    coordinate: np.ndarray | None = None


def _det_keypoints(row: dict, *, side: str) -> np.ndarray | None:
    key = "det_left_kpts" if side == "left" else "det_right_kpts"
    raw = row.get(key)
    if raw is None:
        return None
    kp = np.asarray(raw, dtype=np.float32)
    if kp.ndim != 2 or kp.shape[1] < 2:
        return None
    return kp


def _mean_det_conf(row: dict) -> float:
    vals: list[float] = []
    for side in ("left", "right"):
        kp = _det_keypoints(row, side=side)
        if kp is None:
            continue
        if kp.shape[1] >= 3:
            vals.extend(float(x) for x in kp[:, 2])
    return float(np.mean(vals)) if vals else 0.0


def _reproj_error_px(
    kpts3d: np.ndarray,
    row: dict,
    calib: StereoCalibration,
    *,
    min_conf: float,
) -> float:
    kl = _det_keypoints(row, side="left")
    kr = _det_keypoints(row, side="right")
    if kl is None and kr is None:
        return 0.0

    k = np.asarray(kpts3d, dtype=np.float64)
    errs: list[float] = []
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    rvec_r, _ = cv2.Rodrigues(calib.R)

    for ji in range(k.shape[0]):
        conf = float(k[ji, 3])
        if conf < min_conf:
            continue
        xyz = k[ji, :3].reshape(1, 1, 3)
        if kl is not None and float(kl[ji, 2] if kl.shape[1] >= 3 else 1.0) >= min_conf:
            pred, _ = cv2.projectPoints(xyz, rvec, tvec, calib.K1, calib.D1)
            du = float(pred[0, 0, 0]) - float(kl[ji, 0])
            dv = float(pred[0, 0, 1]) - float(kl[ji, 1])
            errs.append(float(np.hypot(du, dv)))
        if kr is not None and float(kr[ji, 2] if kr.shape[1] >= 3 else 1.0) >= min_conf:
            pred, _ = cv2.projectPoints(xyz, rvec_r, calib.T, calib.K2, calib.D2)
            du = float(pred[0, 0, 0]) - float(kr[ji, 0])
            dv = float(pred[0, 0, 1]) - float(kr[ji, 1])
            errs.append(float(np.hypot(du, dv)))
    return float(np.mean(errs)) if errs else 0.0


def _bone_lengths(kpts3d: np.ndarray, *, min_conf: float) -> dict[tuple[int, int], float]:
    k = np.asarray(kpts3d, dtype=np.float64)
    out: dict[tuple[int, int], float] = {}
    for i, j in _BONE_PAIRS:
        if float(k[i, 3]) < min_conf or float(k[j, 3]) < min_conf:
            continue
        out[(i, j)] = float(np.linalg.norm(k[i, :3] - k[j, :3]))
    return out


def _bone_outlier_ratio(
    lengths: dict[tuple[int, int], float],
    ref: dict[tuple[int, int], float],
    *,
    tol: float,
) -> float:
    if not lengths or not ref:
        return 0.0
    bad = 0
    total = 0
    for key, ln in lengths.items():
        if key not in ref or ref[key] <= 1e-6:
            continue
        total += 1
        ratio = ln / ref[key]
        if ratio < (1.0 - tol) or ratio > (1.0 + tol):
            bad += 1
    return float(bad / total) if total else 0.0


def _max_joint_speed(
    kpts3d: np.ndarray,
    prev: np.ndarray | None,
    *,
    min_conf: float,
    fps: float,
) -> float:
    if prev is None:
        return 0.0
    k = np.asarray(kpts3d, dtype=np.float64)
    p = np.asarray(prev, dtype=np.float64)
    dt = 1.0 / max(float(fps), 1e-6)
    speeds: list[float] = []
    for ji in range(k.shape[0]):
        if float(k[ji, 3]) < min_conf or float(p[ji, 3]) < min_conf:
            continue
        speeds.append(float(np.linalg.norm(k[ji, :3] - p[ji, :3]) / dt))
    return float(max(speeds)) if speeds else 0.0


def score_pose_frame(
    row: dict,
    *,
    cfg: QcConfig,
    calib: StereoCalibration | None = None,
    bone_ref: dict[tuple[int, int], float] | None = None,
    prev_kpts3d: np.ndarray | None = None,
) -> QcFrameResult:
    flags: list[str] = []
    tri_ok = bool(row.get("tri_ok", False))
    raw = row.get("kpts3d")
    kpts3d = np.asarray(raw, dtype=np.float32) if raw is not None else None

    if cfg.require_tri_ok and not tri_ok:
        flags.append("tri_fail")
    if kpts3d is None or kpts3d.size == 0:
        flags.append("no_kpts3d")
        return QcFrameResult(qc_score=1.0, qc_pass=False, qc_flags=flags)

    valid_mask = (kpts3d[:, 3] >= cfg.min_conf) & (
        np.linalg.norm(kpts3d[:, :3], axis=1) > 1e-4
    )
    valid_count = int(valid_mask.sum())
    if valid_count < cfg.min_valid_joints:
        flags.append("low_valid_joints")

    core_z = kpts3d[valid_mask, 2] if valid_mask.any() else np.array([], dtype=np.float32)
    if core_z.size and (float(core_z.min()) < cfg.min_z_m or float(core_z.max()) > cfg.max_z_m):
        flags.append("depth_range")
    if valid_mask.any() and float(np.max(kpts3d[valid_mask, 2])) > cfg.max_joint_z_m:
        flags.append("depth_outlier")

    mean_conf = float(kpts3d[valid_mask, 3].mean()) if valid_mask.any() else 0.0
    det_conf = _mean_det_conf(row)
    conf_term = 1.0 - min(1.0, 0.5 * (mean_conf + det_conf))

    valid_term = 1.0 - min(1.0, valid_count / 17.0)

    reproj_px = 0.0
    reproj_term = 0.0
    if calib is not None:
        reproj_px = _reproj_error_px(kpts3d, row, calib, min_conf=cfg.min_conf)
        reproj_term = min(1.0, reproj_px / max(cfg.reproj_px_scale, 1e-6))
        if reproj_px > cfg.reproj_px_scale * 1.2:
            flags.append("high_reproj")

    lengths = _bone_lengths(kpts3d, min_conf=cfg.min_conf)
    bone_ratio = 0.0
    if bone_ref:
        bone_ratio = _bone_outlier_ratio(lengths, bone_ref, tol=cfg.bone_ratio_tol)
        if bone_ratio > 0.35:
            flags.append("bone_outlier")

    max_speed = _max_joint_speed(
        kpts3d, prev_kpts3d, min_conf=cfg.min_conf, fps=cfg.fps
    )
    vel_term = min(1.0, max_speed / max(cfg.max_bone_vel_m_s, 1e-6))
    if max_speed > cfg.max_bone_vel_m_s:
        flags.append("velocity_spike")

    score = (
        cfg.w_conf * conf_term
        + cfg.w_valid * valid_term
        + cfg.w_reproj * reproj_term
        + cfg.w_bone * bone_ratio
        + cfg.w_velocity * vel_term
    )
    score = float(np.clip(score, 0.0, 1.0))

    hard_fail = (
        (cfg.require_tri_ok and not tri_ok)
        or kpts3d is None
        or valid_count < cfg.min_valid_joints
    )
    qc_pass = (not hard_fail) and score <= cfg.pass_score_threshold

    h36m_kpts, h36m_valid = coco_kpts3d_to_h36m(kpts3d)
    coordinate, _ = h36m_coord_for_export(h36m_kpts)

    return QcFrameResult(
        qc_score=score,
        qc_pass=qc_pass,
        qc_flags=flags,
        mean_conf=mean_conf,
        valid_joint_count=valid_count,
        reproj_px=reproj_px,
        bone_outlier_ratio=bone_ratio,
        max_joint_speed_m_s=max_speed,
        h36m_kpts=h36m_kpts,
        h36m_valid=h36m_valid,
        coordinate=coordinate,
    )
