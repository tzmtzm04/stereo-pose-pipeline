"""流水线数据包与计时。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.infer.pose3d_pipeline import StereoPose3DResult


@dataclass
class FrameTiming:
    capture_ms: float = 0.0
    pull_ms: float = 0.0
    vic_prep_ms: float = 0.0
    infer_parallel_ms: float = 0.0
    infer_egl_ms: float = 0.0
    infer_lb_ms: float = 0.0
    infer_trt_ms: float = 0.0
    infer_post_ms: float = 0.0
    infer_drain_ms: float = 0.0
    infer_wait_ms: float = 0.0
    infer_unmap_ms: float = 0.0
    infer_trt_enqueue_ms: float = 0.0
    infer_trt_d2h_sync_ms: float = 0.0
    triangulate_ms: float = 0.0
    draw_ms: float = 0.0
    preview_ms: float = 0.0
    display_ms: float = 0.0
    queue_wait_ms: float = 0.0
    e2e_ms: float = 0.0
    consumer_ms: float = 0.0

    def from_infer_result(self, result: StereoPose3DResult) -> None:
        self.capture_ms = result.infer.capture_ms
        self.pull_ms = float(getattr(result.infer, "pull_ms", 0.0))
        self.vic_prep_ms = float(getattr(result.infer, "vic_prep_ms", 0.0))
        self.infer_parallel_ms = result.infer.infer_parallel_ms
        self.triangulate_ms = float(getattr(result, "tri_ms", 0.0))
        self.apply_par_breakdown(result.infer.par_breakdown())

    def apply_par_breakdown(self, bd) -> None:
        self.infer_egl_ms = float(bd.egl_ms)
        self.infer_lb_ms = float(bd.lb_ms)
        self.infer_trt_ms = float(bd.trt_ms)
        self.infer_post_ms = float(bd.post_ms)
        self.infer_drain_ms = float(bd.drain_ms)
        self.infer_wait_ms = float(bd.wait_ms)
        self.infer_unmap_ms = float(bd.unmap_ms)
        self.infer_trt_enqueue_ms = float(bd.trt_enqueue_ms)
        self.infer_trt_d2h_sync_ms = float(bd.trt_d2h_sync_ms)

    def format_log_line(self) -> str:
        if self.vic_prep_ms > 0.0:
            cap_tag = f"pull={self.pull_ms:.1f} vic={self.vic_prep_ms:.1f}"
        else:
            cap_tag = f"cap={self.capture_ms:.1f}"
        par_detail = (
            f" egl={self.infer_egl_ms:.1f} lb={self.infer_lb_ms:.1f} "
            f"trt={self.infer_trt_ms:.1f} post={self.infer_post_ms:.1f}"
            f" drain={self.infer_drain_ms:.1f} wait={self.infer_wait_ms:.1f} "
            f"unmap={self.infer_unmap_ms:.1f} trt_eq={self.infer_trt_enqueue_ms:.1f} "
            f"trt_sync={self.infer_trt_d2h_sync_ms:.1f}"
        )
        geo = ""
        if self.triangulate_ms > 0.0:
            geo = f" tri={self.triangulate_ms:.1f}"
        return (
            f"{cap_tag} par={self.infer_parallel_ms:.1f}{par_detail}{geo} "
            f"prev={self.preview_ms:.1f} draw={self.draw_ms:.1f} dsp={self.display_ms:.1f}"
        )


@dataclass
class DisplayPacket:
    pose: StereoPose3DResult
    timing: FrameTiming = field(default_factory=FrameTiming)
    kpts3d: np.ndarray | None = None
    preview_bgr: np.ndarray | None = None
    sbs: Any | None = None
    _preview_keepalive: Any = None

    @property
    def capture_seq(self) -> int:
        return int(self.pose.infer.capture_seq)

    @property
    def infer_seq(self) -> int:
        return int(self.pose.infer.frame_id)
