"""COCO-17 髋角（肩-髋-膝）序列，供预览。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.offline.hip_velocity_smooth import resolve_smooth_params, smooth_series
from core.offline.offline_preview import OfflinePoseRow

_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_KNEE = 13
_RIGHT_KNEE = 14


@dataclass(frozen=True)
class HipAngleSeries:
    capture_seqs: np.ndarray
    angle_raw: np.ndarray
    angle_smooth: np.ndarray
    left_raw: np.ndarray
    right_raw: np.ndarray
    smooth_level: str = "medium"


def hip_joint_angle_deg(kpts: np.ndarray, *, side: str = "left") -> float:
    """单髋肩-髋-膝夹角（度），越大越接近站立伸展。"""
    if side == "left":
        sh, hip, knee = _LEFT_SHOULDER, _LEFT_HIP, _LEFT_KNEE
    else:
        sh, hip, knee = _RIGHT_SHOULDER, _RIGHT_HIP, _RIGHT_KNEE
    if kpts is None or kpts.shape[0] <= knee:
        return float("nan")
    pts = kpts[[sh, hip, knee], :3]
    if not np.isfinite(pts).all():
        return float("nan")
    u = pts[0] - pts[1]
    v = pts[2] - pts[1]
    nu = float(np.linalg.norm(u))
    nv = float(np.linalg.norm(v))
    if nu < 1e-6 or nv < 1e-6:
        return float("nan")
    c = float(np.clip(float(np.dot(u, v) / (nu * nv)), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def bilateral_hip_angle_deg(kpts: np.ndarray) -> tuple[float, float, float]:
    """返回 (mean, left, right)。"""
    la = hip_joint_angle_deg(kpts, side="left")
    ra = hip_joint_angle_deg(kpts, side="right")
    vals = [x for x in (la, ra) if np.isfinite(x)]
    if not vals:
        return float("nan"), la, ra
    return float(np.mean(vals)), la, ra


def compute_hip_angle_series(
    poses: list[OfflinePoseRow],
    fps: float,
    *,
    smooth_level: str = "medium",
    smooth_win: int | None = None,
    smooth_method: str | None = None,
    smooth_passes: int | None = None,
) -> HipAngleSeries:
    seqs: list[int] = []
    raw: list[float] = []
    left: list[float] = []
    right: list[float] = []
    for row in poses:
        seqs.append(int(row.capture_seq))
        if not row.tri_ok or row.kpts3d is None:
            raw.append(float("nan"))
            left.append(float("nan"))
            right.append(float("nan"))
            continue
        m, la, ra = bilateral_hip_angle_deg(row.kpts3d)
        raw.append(m)
        left.append(la)
        right.append(ra)

    arr = np.asarray(raw, dtype=np.float64)
    level = str(smooth_level or "medium").lower()
    if level in ("none", "off", "raw"):
        smooth = arr.copy()
        tag = "none"
    else:
        params = resolve_smooth_params(
            smooth_level=smooth_level,
            smooth_win=smooth_win,
            smooth_method=smooth_method,
            smooth_passes=smooth_passes,
        )
        smooth = smooth_series(
            arr,
            win=int(params["smooth_win"]),
            method=str(params["smooth_method"]),
            poly_order=int(params.get("poly_order", 3)),
            passes=int(params.get("smooth_passes", 1)),
        )
        tag = str(params.get("smooth_level", smooth_level))
    return HipAngleSeries(
        capture_seqs=np.asarray(seqs, dtype=np.int64),
        angle_raw=arr,
        angle_smooth=smooth,
        left_raw=np.asarray(left, dtype=np.float64),
        right_raw=np.asarray(right, dtype=np.float64),
        smooth_level=tag,
    )
