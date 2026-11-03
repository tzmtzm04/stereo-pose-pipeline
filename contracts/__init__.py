"""跨层数据契约（零业务逻辑）。"""

from .nv12_layout import (
    DEFAULT_SBS_FPS,
    DEFAULT_SBS_HEIGHT,
    DEFAULT_SBS_WIDTH,
    nv12_bytes_per_frame,
    raw_ring_memory_mb_for_seconds,
    raw_ring_target_frames,
)
from .segment import ClipSpec, Segment

__all__ = [
    "ClipSpec",
    "Segment",
    "DEFAULT_SBS_WIDTH",
    "DEFAULT_SBS_HEIGHT",
    "DEFAULT_SBS_FPS",
    "nv12_bytes_per_frame",
    "raw_ring_target_frames",
    "raw_ring_memory_mb_for_seconds",
]
