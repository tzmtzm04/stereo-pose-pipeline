"""NV12 ↔ BGR 工具（CPU 平面读写）。"""

from __future__ import annotations

import cv2
import numpy as np


def nv12_er_planes_to_bgr(
    y: np.ndarray,
    uv: np.ndarray,
    height: int,
    width: int,
    pitch: int | None = None,
) -> np.ndarray:
    h, w = int(height), int(width)
    p = int(pitch if pitch is not None else w)
    y = np.ascontiguousarray(y[:h, :w])
    uv = np.asarray(uv)
    if uv.ndim == 3 and uv.shape[2] == 2:
        uv_line = np.ascontiguousarray(uv.reshape(h // 2, w))
    elif uv.ndim == 2 and uv.shape[1] >= p:
        uv_line = np.ascontiguousarray(uv[: h // 2, :w])
    else:
        uv_line = np.ascontiguousarray(uv.reshape(h // 2, w))
    nv12 = np.empty((h * 3 // 2, w), dtype=np.uint8)
    nv12[:h] = y
    nv12[h:] = uv_line
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2BGR_NV12)


def nv12_er_planes_to_rgb(
    y: np.ndarray, uv: np.ndarray, height: int, width: int
) -> np.ndarray:
    """NV12_ER 平面 → RGB uint8 HWC（供姿态 TRT 预处理，跳过 BGR 中间格式）。"""
    h, w = int(height), int(width)
    y = np.ascontiguousarray(y[:h, :w])
    uv = np.asarray(uv)
    if uv.ndim == 3 and uv.shape[2] == 2:
        uv_line = np.ascontiguousarray(uv.reshape(h // 2, w))
    elif uv.ndim == 2 and uv.shape[1] >= w:
        uv_line = np.ascontiguousarray(uv[: h // 2, :w])
    else:
        uv_line = np.ascontiguousarray(uv.reshape(h // 2, w))
    nv12 = np.empty((h * 3 // 2, w), dtype=np.uint8)
    nv12[:h] = y
    nv12[h:] = uv_line
    return cv2.cvtColor(nv12, cv2.COLOR_YUV2RGB_NV12)


def bgr_to_nv12_planes(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = bgr.shape[:2]
    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    y = np.ascontiguousarray(yuv[:h, :w])
    u = yuv[h : h + h // 4].reshape(h // 2, w // 2)
    v = yuv[h + h // 4 : h + h // 2].reshape(h // 2, w // 2)
    uv = np.empty((h // 2, w), dtype=np.uint8)
    uv[:, 0::2] = u
    uv[:, 1::2] = v
    return y, uv
