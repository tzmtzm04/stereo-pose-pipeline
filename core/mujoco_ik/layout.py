"""WB133 → MuJoCo IK site 映射（pelvis + 髋膝肩肘腕踝，13 点）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from core.offline.two_stage.coco_wholebody_layout import NUM_KEYPOINTS_BODY, NUM_KEYPOINTS_BODY_FEET

DerivedKind = Literal["mid_hips", "mid_shoulders"]


@dataclass(frozen=True)
class IKSiteSpec:
    site_name: str
    wb_index: int | None = None
    derived: DerivedKind | None = None
    weight: float = 1.0


# pelvis + 髋膝肩肘腕踝（13 site，全部进 IK 目标）
IK_SITE_SPECS: tuple[IKSiteSpec, ...] = (
    IKSiteSpec("pelvis", derived="mid_hips", weight=2.5),
    IKSiteSpec("left_shoulder", wb_index=5, weight=1.2),
    IKSiteSpec("left_elbow", wb_index=7, weight=1.0),
    IKSiteSpec("left_wrist", wb_index=9, weight=1.0),
    IKSiteSpec("right_shoulder", wb_index=6, weight=1.2),
    IKSiteSpec("right_elbow", wb_index=8, weight=1.0),
    IKSiteSpec("right_wrist", wb_index=10, weight=1.0),
    IKSiteSpec("left_hip", wb_index=11, weight=1.5),
    IKSiteSpec("left_knee", wb_index=13, weight=1.3),
    IKSiteSpec("left_ankle", wb_index=15, weight=1.2),
    IKSiteSpec("right_hip", wb_index=12, weight=1.5),
    IKSiteSpec("right_knee", wb_index=14, weight=1.3),
    IKSiteSpec("right_ankle", wb_index=16, weight=1.2),
)

SITE_NAMES: tuple[str, ...] = tuple(s.site_name for s in IK_SITE_SPECS)


def _point_valid(kpts: np.ndarray, idx: int, min_conf: float) -> bool:
    if idx < 0 or idx >= kpts.shape[0]:
        return False
    if float(kpts[idx, 3]) < min_conf:
        return False
    xyz = kpts[idx, :3]
    return bool(np.all(np.isfinite(xyz)) and np.linalg.norm(xyz) > 1e-4)


def _derived_mid(
    kpts: np.ndarray,
    i_a: int,
    i_b: int,
    min_conf: float,
) -> np.ndarray | None:
    ok_a = _point_valid(kpts, i_a, min_conf)
    ok_b = _point_valid(kpts, i_b, min_conf)
    if ok_a and ok_b:
        return 0.5 * (kpts[i_a, :3] + kpts[i_b, :3])
    if ok_a:
        return kpts[i_a, :3].copy()
    if ok_b:
        return kpts[i_b, :3].copy()
    return None


def wb133_to_ik_targets(
    kpts3d: np.ndarray,
    *,
    min_conf: float = 0.35,
) -> dict[str, np.ndarray | None]:
    """(133,4) / (23,4) / (17,4) → site_name → 目标 xyz（无效则 None）。"""
    k = np.asarray(kpts3d, dtype=np.float64).reshape(-1, 4)
    if k.shape[0] < NUM_KEYPOINTS_BODY:
        out: dict[str, np.ndarray | None] = {name: None for name in SITE_NAMES}
        return out

    out: dict[str, np.ndarray | None] = {}
    for spec in IK_SITE_SPECS:
        if spec.derived == "mid_hips":
            out[spec.site_name] = _derived_mid(k, 11, 12, min_conf)
        elif spec.derived == "mid_shoulders":
            out[spec.site_name] = _derived_mid(k, 5, 6, min_conf)
        elif spec.wb_index is not None:
            if _point_valid(k, spec.wb_index, min_conf):
                out[spec.site_name] = k[spec.wb_index, :3].copy()
            else:
                out[spec.site_name] = None
        else:
            out[spec.site_name] = None
    return out


def site_weights() -> dict[str, float]:
    return {s.site_name: float(s.weight) for s in IK_SITE_SPECS}


def apply_ik_targets_to_kpts3d(
    kpts3d: np.ndarray,
    targets: dict[str, np.ndarray | None],
) -> np.ndarray:
    """将 13 site 硬投影结果写回 body_feet 对应 kpts3d 索引（不含 derived pelvis）。"""
    k = np.asarray(kpts3d, dtype=np.float64).copy().reshape(-1, 4)
    for spec in IK_SITE_SPECS:
        if spec.wb_index is None:
            continue
        t = targets.get(spec.site_name)
        if t is None:
            continue
        xyz = np.asarray(t, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(xyz)):
            continue
        k[spec.wb_index, :3] = xyz
        if float(k[spec.wb_index, 3]) <= 0.0:
            k[spec.wb_index, 3] = 1.0
    return k


def describe_layout() -> str:
    sites = ", ".join(SITE_NAMES)
    return (
        f"COCO-17 / body_feet IK：pelvis + 髋膝肩肘腕踝（13 site）；"
        f"输入需 COCO body 前 {NUM_KEYPOINTS_BODY} 点（或 body+feet {NUM_KEYPOINTS_BODY_FEET}）。"
        f" sites=[{sites}]"
    )
