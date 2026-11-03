"""Path 6：COCO-17 tri marker → MSK MJCF site（{Marker}_marker）。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.mujoco_ik.layout import IKSiteSpec
from core.pose2sim.coco17_layout import COCO17_P2S_MARKERS

DEFAULT_SITE_SUFFIX = "_marker"

# marker 名 → COCO-17 kpts3d 索引
MARKER_COCO_INDEX: dict[str, int] = {name: idx for name, idx in COCO17_P2S_MARKERS}

MARKER_WEIGHTS: dict[str, float] = {
    "Nose": 1.0,
    "RShoulder": 1.2,
    "RElbow": 1.0,
    "RWrist": 1.0,
    "LShoulder": 1.2,
    "LElbow": 1.0,
    "LWrist": 1.0,
    "RHip": 1.5,
    "RKnee": 1.3,
    "RAnkle": 1.2,
    "LHip": 1.5,
    "LKnee": 1.3,
    "LAnkle": 1.2,
}


def _point_valid(kpts: np.ndarray, idx: int, min_conf: float) -> bool:
    if idx < 0 or idx >= kpts.shape[0]:
        return False
    if float(kpts[idx, 3]) < min_conf:
        return False
    xyz = kpts[idx, :3]
    return bool(np.all(np.isfinite(xyz)) and np.linalg.norm(xyz) > 1e-4)


def load_path6_marker_map(path: Path | str) -> tuple[str, ...]:
    data = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    markers = data.get("markers")
    if not markers:
        raise ValueError(f"marker map 缺少 markers: {path}")
    return tuple(str(m) for m in markers)


def marker_to_site_name(marker: str, *, suffix: str = DEFAULT_SITE_SUFFIX) -> str:
    return f"{marker}{suffix}"


def build_path6_site_specs(
    markers: tuple[str, ...] | list[str],
    *,
    suffix: str = DEFAULT_SITE_SUFFIX,
) -> tuple[IKSiteSpec, ...]:
    specs: list[IKSiteSpec] = []
    for marker in markers:
        idx = MARKER_COCO_INDEX.get(marker)
        if idx is None:
            raise ValueError(f"未知 path6 marker（无 COCO 索引）: {marker}")
        site = marker_to_site_name(marker, suffix=suffix)
        weight = float(MARKER_WEIGHTS.get(marker, 1.0))
        specs.append(IKSiteSpec(site, wb_index=idx, weight=weight))
    return tuple(specs)


def path6_site_names(markers: tuple[str, ...] | list[str], *, suffix: str = DEFAULT_SITE_SUFFIX) -> tuple[str, ...]:
    return tuple(marker_to_site_name(m, suffix=suffix) for m in markers)


def path6_to_ik_targets(
    kpts3d: np.ndarray,
    *,
    site_specs: tuple[IKSiteSpec, ...],
    min_conf: float = 0.35,
) -> dict[str, np.ndarray | None]:
    """(17,4) 或 (133,4) → site_name → 目标 xyz。"""
    k = np.asarray(kpts3d, dtype=np.float64).reshape(-1, 4)
    out: dict[str, np.ndarray | None] = {}
    for spec in site_specs:
        if spec.wb_index is None:
            out[spec.site_name] = None
            continue
        if _point_valid(k, spec.wb_index, min_conf):
            out[spec.site_name] = k[spec.wb_index, :3].copy()
        else:
            out[spec.site_name] = None
    return out


def describe_path6_layout(markers: tuple[str, ...]) -> str:
    sites = ", ".join(path6_site_names(markers))
    return f"Path6 MSK IK：{len(markers)} marker sites=[{sites}]"
