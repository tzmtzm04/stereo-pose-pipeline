"""录像环模式常量与归一化。"""

from __future__ import annotations

RING_MODE_SESSION = "session"
RING_MODE_RAW_NV12 = "raw_nv12"
RING_MODE_OFF = "off"

_ALIASES = {
    "session_h264": RING_MODE_SESSION,
    "session_h264_grab": RING_MODE_SESSION,
}


def normalize_ring_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    return _ALIASES.get(key, key)


def is_session_mode(mode: str) -> bool:
    return normalize_ring_mode(mode) == RING_MODE_SESSION


def is_raw_nv12_mode(mode: str) -> bool:
    return normalize_ring_mode(mode) == RING_MODE_RAW_NV12
