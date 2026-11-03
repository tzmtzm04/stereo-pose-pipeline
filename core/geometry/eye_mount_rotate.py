"""竖装双目：左右眼同向 90° 转正语义。

物理约定：整机 90°、左相机在上、右相机在下时，缓冲仍是水平 SBS。
- infer：VIC 裁半与 Rotate90 同次完成 → YOLO 吃竖图，关键点为竖图坐标。
- 预览：display producer 仍出横幅；live_jpeg 对自有 BGR CPU 转正后再叠骨架。
反了用 VISION_EYE_ROTATE=ccw90。
竖装 3D：加载标定后经 ``adapt_stereo_calib_for_eye_rotate`` 把 K/R/T/D
变到竖图像素系（D 按归一化平面 90° 严格变换；也可用 zero_distortion 置零）。
"""

from __future__ import annotations

import os
import threading
from typing import Literal

import numpy as np

EyeMountRotate = Literal["none", "cw90", "ccw90"]

_VALID: frozenset[str] = frozenset(("none", "cw90", "ccw90"))
_rotate_lock = threading.Lock()


def parse_eye_mount_rotate(value: str | None) -> EyeMountRotate:
    raw = (value or "none").strip().lower()
    if raw in ("", "off", "0", "false", "no"):
        return "none"
    if raw in ("cw", "90", "cw90", "clockwise"):
        return "cw90"
    if raw in ("ccw", "ccw90", "anticlockwise", "counterclockwise"):
        return "ccw90"
    if raw in _VALID:
        return raw  # type: ignore[return-value]
    raise ValueError(f"eye_rotate 须 none|cw90|ccw90，收到 {value!r}")


def eye_mount_rotate_from_env(
    *,
    env_key: str = "VISION_EYE_ROTATE",
    default: EyeMountRotate = "none",
) -> EyeMountRotate:
    raw = os.environ.get(env_key)
    if raw is None or not str(raw).strip():
        return default
    return parse_eye_mount_rotate(raw)


def rotated_eye_hw(height: int, width: int, mode: EyeMountRotate) -> tuple[int, int]:
    """返回旋转后 (height, width)。"""
    if mode == "none":
        return int(height), int(width)
    return int(width), int(height)


def apply_eye_mount_rotate(bgr: np.ndarray, mode: EyeMountRotate) -> np.ndarray:
    """对单眼 BGR 做 90° 旋转；mode=none 时原样返回（不拷贝）。

    深拷贝 + 与 VIC 全局锁串行：避免 NVMM/VIC 与 CPU 旋转并发段错误。
    """
    if mode == "none":
        return bgr
    # 强制脱离任何 mmap / 视图（Jetson NV12→BGR 偶发非自有内存）
    src = np.array(bgr, dtype=np.uint8, copy=True, order="C")
    if src.ndim != 3 or src.shape[2] < 3:
        raise ValueError(f"eye rotate 期望 HxWx3，收到 shape={src.shape}")
    from platform.jetson.nvbuf_vic_scale import vic_op_lock

    with vic_op_lock(), _rotate_lock:
        if mode == "cw90":
            # 顺时针 90°：(i,j) → (j, H-1-i)
            out = np.ascontiguousarray(src.transpose(1, 0, 2)[:, ::-1])
        else:
            # 逆时针 90°：(i,j) → (W-1-j, i)
            out = np.ascontiguousarray(src.transpose(1, 0, 2)[::-1, :])
    return out
