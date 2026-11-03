"""H.264 byte-stream 辅助（编码环 keyframe 判定）。"""

from __future__ import annotations


def _iter_nal_types(data: bytes):
    i = 0
    n = len(data)
    while i < n - 3:
        if data[i : i + 4] == b"\x00\x00\x00\x01":
            i += 4
            if i < n:
                yield data[i] & 0x1F
            continue
        if data[i : i + 3] == b"\x00\x00\x01":
            i += 3
            if i < n:
                yield data[i] & 0x1F
            continue
        i += 1


def is_h264_keyframe_au(data: bytes) -> bool:
    if not data:
        return False
    has_sps = False
    for nal_type in _iter_nal_types(data):
        if nal_type == 5:
            return True
        if nal_type == 7:
            has_sps = True
    return has_sps


def packet_is_keyframe(
    data: bytes,
    *,
    is_keyframe_flag: bool = False,
    capture_seq: int | None = None,
    iframe_interval: int | None = None,
) -> bool:
    if data and b"\x00\x00\x01" in data:
        if is_h264_keyframe_au(data):
            return True
        if capture_seq is not None and iframe_interval is not None:
            g = max(1, int(iframe_interval))
            return int(capture_seq) % g == 0
        return False
    return is_keyframe_flag
