"""GL 2D 骨架 overlay（与 pose2d_overlay 同色/拓扑）。"""

from __future__ import annotations

import numpy as np

from app.route2.session_config import Route2SessionConfig
from core.display.display_eye import select_pose_det
from core.display.pose_store import PoseDisplaySample
from platform.jetson.nv12_gl_display import Nv12GlDisplay, Route2GlKpt


def apply_gl_pose_overlay(
    gl: Nv12GlDisplay,
    pose: PoseDisplaySample | None,
    cfg: Route2SessionConfig,
    *,
    video_w: int,
    video_h: int,
) -> None:
    """将 match 到的 pose 写入 GL overlay（camera_overlay 关则清空）。"""
    det = select_pose_det(pose, cfg)
    if not cfg.camera_overlay or det is None:
        gl.clear_pose()
        return

    kpts = det.get("keypoints")
    if kpts is None or len(kpts) == 0:
        gl.clear_pose()
        return

    kpts = np.asarray(kpts, dtype=np.float32)
    kpts3d = pose.kpts3d
    conf = float(cfg.conf)
    out: list[Route2GlKpt] = []
    for ki in range(min(int(kpts.shape[0]), 17)):
        has_depth = 0
        if (
            kpts3d is not None
            and ki < kpts3d.shape[0]
            and kpts3d[ki, 3] >= conf
            and kpts3d[ki, 2] > 0
        ):
            has_depth = 1
        out.append(
            Route2GlKpt(
                x=float(kpts[ki, 0]),
                y=float(kpts[ki, 1]),
                conf=float(kpts[ki, 2]),
                has_depth=has_depth,
            )
        )

    gl.set_pose(
        out,
        conf_thresh=conf,
        video_w=int(video_w),
        video_h=int(video_h),
    )
