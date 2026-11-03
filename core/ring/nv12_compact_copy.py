"""NVMM Y/UV planes → 紧凑 NV12 bytes。"""

from __future__ import annotations

import numpy as np


def nv12_compact_bytes_from_planes(
    y_plane: np.ndarray,
    uv_plane: np.ndarray,
    w: int,
    h: int,
    pitch: int,
    *,
    scratch: bytearray | None = None,
) -> bytes:
    """
    NVMM mapped Y/UV planes → 紧凑 NV12（仅有效宽 w，不含 pitch padding）。

    使用 NumPy bulk copy，避免逐行 Python tobytes()。
    """
    w_i, h_i = int(w), int(h)
    nbytes = w_i * h_i * 3 // 2
    y_view = y_plane[:, :w_i]
    uv_view = uv_plane[:, :w_i]
    if (
        int(pitch) == w_i
        and y_view.flags["C_CONTIGUOUS"]
        and uv_view.flags["C_CONTIGUOUS"]
    ):
        y_part = y_view.ravel()
        uv_part = uv_view.ravel()
    else:
        y_part = np.ascontiguousarray(y_view).ravel()
        uv_part = np.ascontiguousarray(uv_view).ravel()

    if scratch is not None and len(scratch) == nbytes:
        y_dst = np.frombuffer(scratch, dtype=np.uint8, count=w_i * h_i)
        uv_dst = np.frombuffer(
            scratch,
            dtype=np.uint8,
            offset=w_i * h_i,
            count=w_i * h_i // 2,
        )
        np.copyto(y_dst, y_part)
        np.copyto(uv_dst, uv_part)
        return bytes(scratch)
    return y_part.tobytes() + uv_part.tobytes()
