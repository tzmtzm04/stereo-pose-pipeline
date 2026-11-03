"""OpenCap TRC → 3D 骨架预览。"""

from __future__ import annotations

import numpy as np

from core.display.skeleton3d_view import Skeleton3DRotation, render_skeleton3d_bgr
from core.opencap.opencap_skeleton_layout import (
    LAYOUT_OPENCAP_AUG,
    OPENCAP_CENTER_INDICES,
    OpencapSkeletonSpec,
    resolve_opencap_skeleton_edges,
)
from core.opencap.trc_io import TrcData, yup_to_left_cam_opencv


def trc_frame_to_kpts4d(trc: TrcData, frame_idx: int) -> np.ndarray:
    """单帧 (M,4) 左相机 OpenCV 米制；conf=1 表示有效，NaN→conf=0。"""
    i = int(np.clip(frame_idx, 0, trc.coords_yup.shape[0] - 1))
    xyz = yup_to_left_cam_opencv(trc.coords_yup[i])
    conf = np.ones((xyz.shape[0], 1), dtype=np.float32)
    finite = np.all(np.isfinite(xyz), axis=1)
    conf[~finite] = 0.0
    xyz = np.where(finite[:, None], xyz, 0.0)
    return np.concatenate([xyz, conf], axis=1)


def compose_opencap_skeleton_frame(
    trc: TrcData | None,
    frame_idx: int,
    *,
    rotation: Skeleton3DRotation,
    spec: OpencapSkeletonSpec | None = None,
    width: int = 1280,
    height: int = 960,
    label: str | None = None,
) -> tuple[np.ndarray, str]:
    if trc is None or trc.coords_yup.shape[0] == 0:
        canvas = render_skeleton3d_bgr(
            None,
            width=width,
            height=height,
            conf=0.25,
            rotation=rotation,
            label="no TRC",
        )
        return canvas, "no TRC"

    if spec is None:
        spec = resolve_opencap_skeleton_edges(trc.marker_names)
    kpts = trc_frame_to_kpts4d(trc, frame_idx)
    canvas = render_skeleton3d_bgr(
        kpts,
        width=width,
        height=height,
        conf=0.25,
        rotation=rotation,
        ignore_conf=True,
        skeleton_edges=spec.edges,
        layout=LAYOUT_OPENCAP_AUG,
        label=label or spec.label,
        bone_color=(0, 200, 255),
        marker_names=trc.marker_names,
        center_indices=OPENCAP_CENTER_INDICES,
    )
    return canvas, spec.label
