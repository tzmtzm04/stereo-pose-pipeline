"""3D 骨架预览布局：COCO-17 / body+feet / WholeBody-133。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from core.display.skeleton3d_view import COCO_SKELETON_17
from core.offline.two_stage.coco_wholebody_layout import (
    BODY_FEET_SKELETON,
    NUM_KEYPOINTS_BODY,
    NUM_KEYPOINTS_BODY_FEET,
    NUM_KEYPOINTS_WHOLEBODY,
    WHOLEBODY_FACE_IDX_END,
    WHOLEBODY_FACE_IDX_START,
    WHOLEBODY_HAND_LEFT_IDX_START,
    WHOLEBODY_SKELETON_EDGES,
)

Skeleton3DLayoutMode = Literal["auto", "coco17", "body_feet", "wholebody"]


@dataclass(frozen=True)
class Skeleton3DRenderSpec:
    layout: str
    edges: list[tuple[int, int]]
    num_joints: int
    label: str


def infer_num_joints(kpts3d: np.ndarray | None) -> int:
    if kpts3d is None:
        return NUM_KEYPOINTS_BODY
    k = np.asarray(kpts3d)
    if k.ndim != 2 or k.shape[1] < 3:
        return NUM_KEYPOINTS_BODY
    return int(k.shape[0])


def resolve_skeleton3d_render_spec(
    kpts3d: np.ndarray | None,
    *,
    mode: Skeleton3DLayoutMode = "auto",
    include_feet: bool = True,
) -> Skeleton3DRenderSpec:
    n = infer_num_joints(kpts3d)
    key = str(mode).strip().lower()

    if key == "coco17":
        return Skeleton3DRenderSpec(
            layout="coco17",
            edges=list(COCO_SKELETON_17),
            num_joints=NUM_KEYPOINTS_BODY,
            label="COCO-17",
        )
    if key == "body_feet":
        if include_feet:
            return Skeleton3DRenderSpec(
                layout="body_feet",
                edges=list(BODY_FEET_SKELETON),
                num_joints=NUM_KEYPOINTS_BODY_FEET,
                label="body+feet(23)",
            )
        return Skeleton3DRenderSpec(
            layout="coco17",
            edges=list(COCO_SKELETON_17),
            num_joints=NUM_KEYPOINTS_BODY,
            label="body(17)",
        )
    if key == "wholebody":
        return Skeleton3DRenderSpec(
            layout="wholebody133",
            edges=list(WHOLEBODY_SKELETON_EDGES),
            num_joints=NUM_KEYPOINTS_WHOLEBODY,
            label="WholeBody-133",
        )

    # auto
    if n >= NUM_KEYPOINTS_WHOLEBODY:
        return Skeleton3DRenderSpec(
            layout="wholebody133",
            edges=list(WHOLEBODY_SKELETON_EDGES),
            num_joints=NUM_KEYPOINTS_WHOLEBODY,
            label="WholeBody-133",
        )
    if n >= NUM_KEYPOINTS_BODY_FEET:
        if include_feet:
            nj = min(n, NUM_KEYPOINTS_BODY_FEET)
            return Skeleton3DRenderSpec(
                layout="body_feet",
                edges=list(BODY_FEET_SKELETON),
                num_joints=nj,
                label=f"body+feet({nj})",
            )
        return Skeleton3DRenderSpec(
            layout="coco17",
            edges=list(COCO_SKELETON_17),
            num_joints=NUM_KEYPOINTS_BODY,
            label="body(17)",
        )
    return Skeleton3DRenderSpec(
        layout="coco17",
        edges=list(COCO_SKELETON_17),
        num_joints=min(n, NUM_KEYPOINTS_BODY),
        label="COCO-17",
    )


def joint_style_for_layout(
    joint_idx: int,
    layout: str,
    *,
    dimmed: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    """返回 (radius, BGR)。"""
    if layout == "wholebody133":
        if WHOLEBODY_FACE_IDX_START <= joint_idx < WHOLEBODY_FACE_IDX_END:
            return 2, (150, 150, 170) if dimmed else (190, 190, 210)
        if joint_idx >= WHOLEBODY_HAND_LEFT_IDX_START:
            return 3, (90, 150, 220) if dimmed else (120, 200, 255)
        if joint_idx >= NUM_KEYPOINTS_BODY:
            return 4, (80, 180, 255) if dimmed else (100, 220, 255)
    if layout == "body_feet" and joint_idx >= NUM_KEYPOINTS_BODY:
        return 4, (80, 180, 255) if dimmed else (100, 220, 255)
    if layout == "soma78":
        from core.display.soma_skeleton3d_layout import joint_style_soma78

        return joint_style_soma78(joint_idx, dimmed=dimmed)
    return 5, (0, 150, 200) if dimmed else (0, 200, 255)
