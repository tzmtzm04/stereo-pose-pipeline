"""L3 固定参数。"""

from __future__ import annotations

from contracts.nv12_layout import (
    DEFAULT_RAW_RING_MEMORY_MB,
    DEFAULT_RAW_RING_SECONDS,
)

HISTORY_SECONDS = 30.0
RAW_RING_SECONDS = DEFAULT_RAW_RING_SECONDS
RAW_RING_MAX_MEMORY_MB = DEFAULT_RAW_RING_MEMORY_MB

# Mock 判段占位边界（capture_seq 闭区间）
MOCK_CLIP_SEGMENTS: tuple[tuple[int, int], ...] = ((30, 90), (120, 150))

DEFAULT_BITRATE = 15_000_000
DEFAULT_IFRAME_INTERVAL = 30
CLIP_OUTPUT_ROOT = "outputs/route2_clips"


def history_max_samples(fps: int) -> int:
    fps_i = max(1, int(fps))
    return max(500, int(HISTORY_SECONDS * fps_i * 1.2))
