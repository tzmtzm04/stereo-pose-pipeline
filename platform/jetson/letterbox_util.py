"""Ultralytics LetterBox 几何（与 augment.LetterBox 默认一致）。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxGeometry:
    """letterbox 到 dst_w×dst_h 的参数。"""

    src_w: int
    src_h: int
    dst_w: int
    dst_h: int
    scale: float
    unpad_w: int
    unpad_h: int
    pad_left: int
    pad_top: int

    @property
    def pad_x_float(self) -> float:
        return float(self.pad_left)

    @property
    def pad_y_float(self) -> float:
        return float(self.pad_top)


def ultralytics_letterbox_geometry(
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
    *,
    scaleup: bool = True,
) -> LetterboxGeometry:
    """与 ultralytics LetterBox(center=True, scale_fill=False) 一致。"""
    sw, sh = int(src_w), int(src_h)
    dw, dh = int(dst_w), int(dst_h)
    r = min(dh / sh, dw / sw)
    if not scaleup:
        r = min(r, 1.0)
    unpad_w = int(round(sw * r))
    unpad_h = int(round(sh * r))
    pad_w = dw - unpad_w
    pad_h = dh - unpad_h
    pad_left = int(round(pad_w / 2 - 0.1))
    pad_top = int(round(pad_h / 2 - 0.1))
    return LetterboxGeometry(
        src_w=sw,
        src_h=sh,
        dst_w=dw,
        dst_h=dh,
        scale=r,
        unpad_w=unpad_w,
        unpad_h=unpad_h,
        pad_left=pad_left,
        pad_top=pad_top,
    )


def letterbox_descale_params(
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> tuple[float, float, float, int, int]:
    """
    与 YOLOs-CPP preprocessing::getScalePad 一致（反变换用 float pad，非 int 边框）。
    返回 scale, pad_x, pad_y, new_w, new_h。
    """
    sw, sh = int(src_w), int(src_h)
    dw, dh = int(dst_w), int(dst_h)
    scale = min(dh / sh, dw / sw)
    new_w = int(round(sw * scale))
    new_h = int(round(sh * scale))
    pad_x = (dw - new_w) / 2.0
    pad_y = (dh - new_h) / 2.0
    return scale, pad_x, pad_y, new_w, new_h


def letterbox_descale_params_from_geometry(geo: LetterboxGeometry) -> tuple[float, float, float, int, int]:
    return letterbox_descale_params(geo.src_w, geo.src_h, geo.dst_w, geo.dst_h)


def letterbox_tensor_to_content(
    x: float,
    y: float,
    geo: LetterboxGeometry,
) -> tuple[float, float]:
    """TRT 640 张量坐标 → unpad 内容区（去掉 letterbox 灰边）。"""
    scale, pad_x, pad_y, _, _ = letterbox_descale_params(
        geo.src_w, geo.src_h, geo.dst_w, geo.dst_h
    )
    return x - pad_x, y - pad_y


def letterbox_content_to_orig(
    cx: float,
    cy: float,
    geo: LetterboxGeometry,
) -> tuple[float, float]:
    """unpad 内容区 → 原图 src_w×src_h。"""
    sx = geo.src_w / float(geo.unpad_w)
    sy = geo.src_h / float(geo.unpad_h)
    return cx * sx, cy * sy


def letterbox_tensor_to_orig(
    x: float,
    y: float,
    geo: LetterboxGeometry,
    *,
    orig_map_mode: str = "float_pad",
) -> tuple[float, float]:
    """
    640 TRT 张量 → 原图（对齐 YOLOs-CPP getScalePad + postprocessV26）。

    orig_map_mode:
      float_pad — float pad（默认，与 Ultralytics / C++ 一致）
      int_pad   — 用 geo.pad_left/pad_top（整数 pad，用于 A/B 验收）
      content   — tensor→content→orig 两步
    """
    mode = str(orig_map_mode or "float_pad").strip().lower()
    if mode in ("float_pad", "float", "default", "yolo_cpp"):
        scale, pad_x, pad_y, _, _ = letterbox_descale_params(
            geo.src_w, geo.src_h, geo.dst_w, geo.dst_h
        )
        inv = 1.0 / float(scale)
        return (x - pad_x) * inv, (y - pad_y) * inv
    if mode in ("int_pad", "int"):
        inv = 1.0 / float(geo.scale)
        return (x - float(geo.pad_left)) * inv, (y - float(geo.pad_top)) * inv
    if mode in ("content", "two_step"):
        cx, cy = letterbox_tensor_to_content(x, y, geo)
        return letterbox_content_to_orig(cx, cy, geo)
    raise ValueError(
        f"未知 orig_map_mode: {orig_map_mode!r}（float_pad | int_pad | content）"
    )


def letterbox_descale_inv(geo: LetterboxGeometry) -> tuple[float, float, float, float]:
    """兼容旧接口：返回 pad_x, pad_y, sx, sy。"""
    pad_x = (geo.dst_w - geo.unpad_w) / 2.0
    pad_y = (geo.dst_h - geo.unpad_h) / 2.0
    sx = geo.src_w / float(geo.unpad_w)
    sy = geo.src_h / float(geo.unpad_h)
    return pad_x, pad_y, sx, sy


def letterbox_bgr_cpu(
    bgr: np.ndarray,
    dst_w: int,
    dst_h: int,
    *,
    scaleup: bool = True,
) -> tuple[np.ndarray, LetterboxGeometry]:
    """CPU BGR LetterBox（与 ultralytics.pre_transform 一致）。"""
    geo = ultralytics_letterbox_geometry(
        bgr.shape[1], bgr.shape[0], dst_w, dst_h, scaleup=scaleup
    )
    resized = cv2.resize(
        bgr,
        (geo.unpad_w, geo.unpad_h),
        interpolation=cv2.INTER_LINEAR,
    )
    pad_w = geo.dst_w - geo.unpad_w
    pad_h = geo.dst_h - geo.unpad_h
    # 与 augment.LetterBox(center=True) 的 round(dh±0.1) 一致
    top = int(round(pad_h / 2 - 0.1))
    bottom = int(round(pad_h / 2 + 0.1))
    left = int(round(pad_w / 2 - 0.1))
    right = int(round(pad_w / 2 + 0.1))
    out = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return out, geo
