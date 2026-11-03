"""NV12 SBS 布局与裸帧环内存预算常量。"""

from __future__ import annotations

import math

DEFAULT_SBS_WIDTH = 3840
DEFAULT_SBS_HEIGHT = 1080
DEFAULT_SBS_FPS = 60
DEFAULT_RAW_RING_SECONDS = 10.0


def nv12_bytes_per_frame(width: int, height: int) -> int:
    return max(1, int(width) * int(height) * 3 // 2)


def raw_ring_target_frames(ring_seconds: float, fps: int) -> int:
    """名义秒数对应帧槽上限（含 20% 帧率余量）。"""
    return max(1, int(float(ring_seconds) * max(1, int(fps)) * 1.2))


def raw_ring_memory_mb_for_seconds(
    ring_seconds: float,
    *,
    width: int = DEFAULT_SBS_WIDTH,
    height: int = DEFAULT_SBS_HEIGHT,
    fps: int = DEFAULT_SBS_FPS,
) -> int:
    max_frames = raw_ring_target_frames(ring_seconds, fps)
    bytes_needed = max_frames * nv12_bytes_per_frame(width, height)
    return max(1, math.ceil(bytes_needed / (1024 * 1024)))


DEFAULT_RAW_RING_MEMORY_MB = raw_ring_memory_mb_for_seconds(DEFAULT_RAW_RING_SECONDS)


def raw_ring_max_frames(
    width: int,
    height: int,
    fps: int,
    ring_seconds: float,
    *,
    max_memory_mb: int = DEFAULT_RAW_RING_MEMORY_MB,
) -> int:
    frame_bytes = nv12_bytes_per_frame(width, height)
    budget = max(1, int(max_memory_mb)) * 1024 * 1024
    from_mem = max(1, budget // frame_bytes)
    from_time = raw_ring_target_frames(ring_seconds, fps)
    return min(from_mem, from_time)
