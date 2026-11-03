"""COCO-17 肩/肘/髋/膝 关节夹角时序（左右同组，单位 deg）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.offline.hip_velocity_smooth import resolve_smooth_params, smooth_series
from core.offline.offline_preview import OfflinePoseRow

_LEFT_SHOULDER = 5
_RIGHT_SHOULDER = 6
_LEFT_ELBOW = 7
_RIGHT_ELBOW = 8
_LEFT_WRIST = 9
_RIGHT_WRIST = 10
_LEFT_HIP = 11
_RIGHT_HIP = 12
_LEFT_KNEE = 13
_RIGHT_KNEE = 14
_LEFT_ANKLE = 15
_RIGHT_ANKLE = 16

JOINT_GROUPS: tuple[tuple[str, tuple[int, int, int], tuple[int, int, int]], ...] = (
    ("shoulder", (_LEFT_HIP, _LEFT_SHOULDER, _LEFT_ELBOW), (_RIGHT_HIP, _RIGHT_SHOULDER, _RIGHT_ELBOW)),
    ("elbow", (_LEFT_SHOULDER, _LEFT_ELBOW, _LEFT_WRIST), (_RIGHT_SHOULDER, _RIGHT_ELBOW, _RIGHT_WRIST)),
    ("hip", (_LEFT_SHOULDER, _LEFT_HIP, _LEFT_KNEE), (_RIGHT_SHOULDER, _RIGHT_HIP, _RIGHT_KNEE)),
    ("knee", (_LEFT_HIP, _LEFT_KNEE, _LEFT_ANKLE), (_RIGHT_HIP, _RIGHT_KNEE, _RIGHT_ANKLE)),
)

_GROUP_ANGLE_HINT = {
    "shoulder": "hip-shoulder-elbow",
    "elbow": "shoulder-elbow-wrist",
    "hip": "shoulder-hip-knee",
    "knee": "hip-knee-ankle",
}


@dataclass(frozen=True)
class JointPairSeries:
    """单关节类型：左右各 raw + 高斯平滑后 (n,) 夹角序列（度）。"""

    left_deg_raw: np.ndarray
    right_deg_raw: np.ndarray
    left_deg: np.ndarray
    right_deg: np.ndarray


@dataclass(frozen=True)
class JointAngleSeries:
    capture_seqs: np.ndarray
    groups: dict[str, JointPairSeries]
    smooth_level: str = "stream"

    @property
    def count(self) -> int:
        return int(len(self.capture_seqs))


def _smooth_angles(arr: np.ndarray, *, smooth_level: str) -> tuple[np.ndarray, str]:
    level = str(smooth_level or "stream").lower()
    if level in ("none", "off", "raw"):
        return np.array(arr, dtype=np.float64, copy=True), "none"
    params = resolve_smooth_params(
        smooth_level=smooth_level,
        smooth_method="gaussian",
    )
    smooth = smooth_series(
        np.asarray(arr, dtype=np.float64),
        win=int(params["smooth_win"]),
        method="gaussian",
        passes=int(params.get("smooth_passes", 1)),
    )
    tag = str(params.get("smooth_level", smooth_level))
    return smooth, tag


def three_point_angle_deg(kpts: np.ndarray | None, i_a: int, i_vertex: int, i_c: int) -> float:
    """顶点 i_vertex 处夹角（度）。"""
    if kpts is None:
        return float("nan")
    hi = max(i_a, i_vertex, i_c)
    if kpts.shape[0] <= hi:
        return float("nan")
    pts = np.asarray(kpts[[i_a, i_vertex, i_c], :3], dtype=np.float64)
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


def compute_joint_angle_series(
    poses: list[OfflinePoseRow],
    *,
    smooth_level: str = "stream",
) -> JointAngleSeries:
    n = len(poses)
    seqs = np.empty(n, dtype=np.int64)
    buckets: dict[str, tuple[list[float], list[float]]] = {
        name: ([], []) for name, _, _ in JOINT_GROUPS
    }

    for i, row in enumerate(poses):
        seqs[i] = int(row.capture_seq)
        kpts = row.kpts3d if row.tri_ok else None
        for name, left_tri, right_tri in JOINT_GROUPS:
            left_list, right_list = buckets[name]
            left_list.append(three_point_angle_deg(kpts, *left_tri))
            right_list.append(three_point_angle_deg(kpts, *right_tri))

    groups: dict[str, JointPairSeries] = {}
    smooth_tag = "none"
    for name, _, _ in JOINT_GROUPS:
        left_list, right_list = buckets[name]
        left_raw = np.asarray(left_list, dtype=np.float64)
        right_raw = np.asarray(right_list, dtype=np.float64)
        left_smooth, tag_l = _smooth_angles(left_raw, smooth_level=smooth_level)
        right_smooth, tag_r = _smooth_angles(right_raw, smooth_level=smooth_level)
        smooth_tag = tag_l if tag_l != "none" else tag_r
        groups[name] = JointPairSeries(
            left_deg_raw=left_raw,
            right_deg_raw=right_raw,
            left_deg=left_smooth,
            right_deg=right_smooth,
        )
    return JointAngleSeries(capture_seqs=seqs, groups=groups, smooth_level=smooth_tag)


# 兼容旧名
JointCoordSeries = JointAngleSeries
compute_joint_coord_series = compute_joint_angle_series
