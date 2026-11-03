"""按 MJCF 选择 IK site 布局与 3D→target 映射（COCO-17 path6 / wb133）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from core.mujoco_ik.layout import IK_SITE_SPECS, IKSiteSpec, describe_layout, wb133_to_ik_targets
from core.mujoco_ik.path6_layout import (
    build_path6_site_specs,
    describe_path6_layout,
    load_path6_marker_map,
    path6_to_ik_targets,
)


@dataclass(frozen=True)
class IKLayoutBundle:
    site_specs: tuple[IKSiteSpec, ...]
    to_ik_targets: Callable[..., dict[str, np.ndarray | None]]
    description: str
    init_from_keyframe: bool = False
    marker_map: Path | None = None
    markers: tuple[str, ...] = ()


def _is_path6_coco17_model(model_path: Path) -> bool:
    name = model_path.name.lower()
    parent = model_path.parent.name.lower()
    return "coco17" in name or (
        "pose2sim_simple" in parent and "wb133" not in name
    )


def resolve_ik_layout(model_path: Path | str) -> IKLayoutBundle:
    """coco17 MSK → path6 markers；wb133_ik_humanoid → 13 site pelvis 派生。"""
    p = Path(model_path).expanduser().resolve()
    if _is_path6_coco17_model(p):
        map_path = p.parent / "marker_map_path6_coco17.json"
        if not map_path.is_file():
            raise FileNotFoundError(f"path6 COCO-17 模型缺少 marker map: {map_path}")
        markers = load_path6_marker_map(map_path)
        specs = build_path6_site_specs(markers)

        def _mapper(
            kpts3d: np.ndarray,
            *,
            min_conf: float = 0.35,
        ) -> dict[str, np.ndarray | None]:
            return path6_to_ik_targets(
                kpts3d,
                site_specs=specs,
                min_conf=min_conf,
            )

        return IKLayoutBundle(
            site_specs=specs,
            to_ik_targets=_mapper,
            description=describe_path6_layout(markers),
            init_from_keyframe=True,
            marker_map=map_path,
            markers=markers,
        )

    return IKLayoutBundle(
        site_specs=IK_SITE_SPECS,
        to_ik_targets=wb133_to_ik_targets,
        description=describe_layout(),
        init_from_keyframe=False,
    )
