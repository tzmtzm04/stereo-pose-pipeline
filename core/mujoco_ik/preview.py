"""MuJoCo IK FK 骨架 → skeleton3d_view 画布。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.display.skeleton3d_view import (
    DEFAULT_VOLUME_5M,
    PerspectiveIntrinsics,
    ProjectionMode,
    Skeleton3DRotation,
    Volume3DBox,
    render_skeleton3d_bgr,
)
from core.mujoco_ik.fk import MujocoFK, MujocoIKFrame
from core.mujoco_ik.skeleton_layout import (
    MUJOCO_IK_CENTER_INDICES,
    MUJOCO_IK_LAYOUT_LABEL,
    MUJOCO_IK_SKELETON_EDGES,
)

_IK_BONE_COLOR = (60, 200, 255)  # BGR 橙
_IK_JOINT_COLOR = (255, 180, 80)


def compose_mujoco_ik_skeleton_frame(
    fk: MujocoFK,
    frame: MujocoIKFrame,
    *,
    rotation: Skeleton3DRotation,
    width: int = 1280,
    height: int = 960,
    projection_mode: ProjectionMode = "volume",
    perspective_intrinsics: PerspectiveIntrinsics | None = None,
    volume_box: Volume3DBox | None = None,
) -> tuple[np.ndarray, str]:
    kpts = fk.site_kpts3d(frame.qpos)
    knees = fk.knee_flex_angles(frame.qpos)
    label = f"{MUJOCO_IK_LAYOUT_LABEL} res={frame.residual:.3f}"
    vol = volume_box or DEFAULT_VOLUME_5M
    if projection_mode == "perspective" and perspective_intrinsics is None:
        perspective_intrinsics = PerspectiveIntrinsics.for_canvas(width, height)
    canvas = render_skeleton3d_bgr(
        kpts,
        width=int(width),
        height=int(height),
        conf=0.25,
        rotation=rotation,
        ignore_conf=True,
        skeleton_edges=MUJOCO_IK_SKELETON_EDGES,
        label=label,
        bone_color=_IK_BONE_COLOR,
        joint_color=_IK_JOINT_COLOR,
        layout="mujoco_ik",
        center_indices=MUJOCO_IK_CENTER_INDICES,
        projection_mode=projection_mode,
        perspective_intrinsics=perspective_intrinsics,
        volume_box=vol,
    )
    cv2.putText(
        canvas,
        knees.overlay_line(),
        (12, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (120, 255, 180),
        1,
        cv2.LINE_AA,
    )
    return canvas, MUJOCO_IK_LAYOUT_LABEL


def default_ik_model_path(root: Path) -> Path:
    from app.paths import resolve_default_mujoco_ik_model

    p = resolve_default_mujoco_ik_model()
    if p.is_file():
        return p
    return root / "models" / "mujoco" / "pose2sim_simple" / "Model_Pose2Sim_simple_coco17_cvt1_compat.xml"
