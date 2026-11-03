"""SBS NV12 → BGR 预览（display 路径，默认左眼）。"""

from __future__ import annotations

import cv2
import numpy as np

from core.geometry.eye_mount_rotate import (
    EyeMountRotate,
    apply_eye_mount_rotate,
    parse_eye_mount_rotate,
    rotated_eye_hw,
)
from platform.jetson.nv12_frame import Nv12Frame, Nv12SbsFrame
from platform.jetson.nv12_util import nv12_er_planes_to_bgr


def _normalize_rotate(eye_rotate: EyeMountRotate | str) -> EyeMountRotate:
    if isinstance(eye_rotate, str):
        return parse_eye_mount_rotate(eye_rotate)
    return eye_rotate


def nv12_fd_to_bgr_cpu(
    fd: int,
    width: int,
    height: int,
    *,
    crop_x: int = 0,
    crop_w: int | None = None,
) -> np.ndarray:
    """NVMM fd 映射平面 → BGR；可只转换水平裁切区域（如 SBS 左眼）。"""
    from platform.jetson.nvbuf_util import (
        NVBUF_MAP_READ,
        nv12_plane_arrays,
        nvbuf_surface_from_fd,
        nvbuf_surface_map,
        nvbuf_surface_sync_for_cpu,
        nvbuf_surface_unmap,
    )

    surf = nvbuf_surface_from_fd(int(fd))
    nvbuf_surface_map(surf, 0, -1, NVBUF_MAP_READ)
    nvbuf_surface_sync_for_cpu(surf, 0, -1)
    try:
        y_plane, uv_plane, w, h, pitch = nv12_plane_arrays(surf, 0)
        cw = int(crop_w if crop_w is not None else w)
        cx = int(crop_x)
        if cx > 0 or cw < w:
            y_plane = np.ascontiguousarray(y_plane[:h, cx : cx + cw])
            uv_plane = np.ascontiguousarray(uv_plane[: h // 2, cx : cx + cw])
            w = cw
            pitch = cw
        return nv12_er_planes_to_bgr(y_plane, uv_plane, h, w, pitch=pitch)
    finally:
        nvbuf_surface_unmap(surf, 0, -1)


def _eye_bgr_from_sbs_full(
    frame: Nv12Frame,
    sbs_width: int,
    *,
    eye: str = "left",
) -> np.ndarray:
    from core.display.display_eye import normalize_display_eye, sbs_crop_x_for_eye

    eye_w = max(1, int(sbs_width) // 2)
    crop_x = sbs_crop_x_for_eye(int(sbs_width), normalize_display_eye(eye))
    if frame.y is not None and frame.uv is not None:
        h = frame.height
        y = np.ascontiguousarray(frame.y[:h, crop_x : crop_x + eye_w])
        uv = np.ascontiguousarray(frame.uv[: h // 2, crop_x : crop_x + eye_w])
        return nv12_er_planes_to_bgr(y, uv, h, eye_w, pitch=eye_w)
    if frame.dmabuf_fd is not None:
        return nv12_fd_to_bgr_cpu(
            frame.dmabuf_fd,
            frame.width,
            frame.height,
            crop_x=crop_x,
            crop_w=eye_w,
        )
    raise RuntimeError("Nv12Frame 无平面也无 fd")


def _left_eye_bgr_from_full(frame: Nv12Frame, sbs_width: int) -> np.ndarray:
    return _eye_bgr_from_sbs_full(frame, sbs_width, eye="left")


def _frame_to_bgr(frame: Nv12Frame) -> np.ndarray:
    if frame.y is not None and frame.uv is not None:
        return frame.to_bgr_cpu()
    if frame.dmabuf_fd is not None:
        return nv12_fd_to_bgr_cpu(frame.dmabuf_fd, frame.width, frame.height)
    raise RuntimeError("Nv12Frame 无平面也无 fd")


def nv12_display_frame_to_preview_bgr(
    frame: Nv12Frame,
    *,
    max_width: int = 0,
    use_cpu: bool = False,
    dst: np.ndarray | None = None,
    display_eye: str = "left",
    eye_rotate: EyeMountRotate | str = "none",
) -> tuple[np.ndarray, float]:
    """display NV12 → BGR；全幅 SBS 默认 GPU CUDA 裁左眼，--cpu-preview 走 CPU。

    dst 非空时 D2H/转换直写 slot 缓冲（P2 少拷贝）；返回 (dst 或新数组, preview_ms)。
    竖装时 eye_rotate 在裁眼后转正；若启用旋转则忽略 dst（尺寸已变）。
    """
    import time

    from core.display.gpu_display_preview import _is_sbs_full_frame

    rotate = _normalize_rotate(eye_rotate)
    if dst is not None and rotate == "none":
        return _nv12_display_to_bgr_into(
            frame,
            dst,
            max_width=max_width,
            use_cpu=use_cpu,
            display_eye=display_eye,
            eye_rotate=rotate,
        )

    t0 = time.perf_counter()
    if use_cpu:
        if _is_sbs_full_frame(frame):
            bgr = _eye_bgr_from_sbs_full(frame, frame.width, eye=display_eye)
        else:
            bgr = nv12_frame_to_preview_bgr(frame, max_width=0)
        bgr = apply_eye_mount_rotate(bgr, rotate)
        bgr = _maybe_resize_bgr(bgr, max_width)
        return bgr, (time.perf_counter() - t0) * 1000.0

    from core.display.gpu_display_preview import (
        gpu_display_preview_available,
        nv12_display_frame_to_preview_bgr as _gpu_preview,
    )

    if not gpu_display_preview_available():
        raise RuntimeError(
            "display GPU preview 不可用，请构建 route2_nv12_rgb.so；"
            "勿对 display 使用 CPU 回退"
        )
    if frame.dmabuf_fd is None:
        raise RuntimeError("display GPU preview 需要 dmabuf_fd")
    bgr, _gpu_ms = _gpu_preview(frame, max_width=0, display_eye=display_eye)
    bgr = apply_eye_mount_rotate(bgr, rotate)
    bgr = _maybe_resize_bgr(bgr, max_width)
    return bgr, (time.perf_counter() - t0) * 1000.0


def display_preview_shape(
    frame: Nv12Frame,
    *,
    eye_rotate: EyeMountRotate | str = "none",
) -> tuple[int, int]:
    """display BGR 输出 (height, width)。"""
    from core.display.gpu_display_preview import preview_eye_shape

    h, w = preview_eye_shape(frame)
    return rotated_eye_hw(h, w, _normalize_rotate(eye_rotate))


def _nv12_display_to_bgr_into(
    frame: Nv12Frame,
    dst: np.ndarray,
    *,
    max_width: int = 0,
    use_cpu: bool = False,
    display_eye: str = "left",
    eye_rotate: EyeMountRotate | str = "none",
) -> tuple[np.ndarray, float]:
    from core.display.gpu_display_preview import (
        _is_sbs_full_frame,
        gpu_display_preview_available,
        nv12_display_frame_to_preview_bgr_into as _gpu_into,
        preview_eye_shape,
    )

    rotate = _normalize_rotate(eye_rotate)
    if rotate != "none":
        return nv12_display_frame_to_preview_bgr(
            frame,
            max_width=max_width,
            use_cpu=use_cpu,
            display_eye=display_eye,
            eye_rotate=rotate,
        )

    h, w = preview_eye_shape(frame)
    if dst.shape != (h, w, 3) or dst.dtype != np.uint8:
        raise ValueError(f"dst 期望 {(h, w, 3)} uint8，收到 {dst.shape} {dst.dtype}")

    if use_cpu:
        import time

        t0 = time.perf_counter()
        if _is_sbs_full_frame(frame):
            bgr = _eye_bgr_from_sbs_full(frame, frame.width, eye=display_eye)
        else:
            bgr = nv12_frame_to_preview_bgr(frame, max_width=0)
        bgr = _maybe_resize_bgr(bgr, max_width)
        np.copyto(dst, bgr)
        return dst, (time.perf_counter() - t0) * 1000.0

    if not gpu_display_preview_available():
        raise RuntimeError(
            "display GPU preview 不可用，请构建 route2_nv12_rgb.so；"
            "勿对 display 使用 CPU 回退"
        )
    if frame.dmabuf_fd is None:
        raise RuntimeError("display GPU preview 需要 dmabuf_fd")
    preview_ms = _gpu_into(frame, dst, max_width=max_width, display_eye=display_eye)
    return dst, preview_ms


def _maybe_resize_bgr(bgr: np.ndarray, max_width: int) -> np.ndarray:
    if max_width > 0 and bgr.shape[1] > max_width:
        h, w = bgr.shape[:2]
        scale = max_width / float(w)
        bgr = cv2.resize(
            bgr,
            (max_width, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return bgr


def nv12_frame_to_preview_bgr(
    frame: Nv12Frame,
    *,
    max_width: int = 0,
) -> np.ndarray:
    """单帧 NV12（单眼或已裁）→ BGR。"""
    bgr = _frame_to_bgr(frame)
    return _maybe_resize_bgr(bgr, max_width)


def sbs_preview_bgr(
    sbs: Nv12SbsFrame,
    *,
    max_width: int = 0,
    eye: str = "left",
    eye_rotate: EyeMountRotate | str = "none",
) -> np.ndarray | None:
    """SBS → 单眼 BGR 预览，可选缩放到 max_width。"""
    if sbs.full is not None:
        bgr = _eye_bgr_from_sbs_full(sbs.full, sbs.width, eye=eye)
    elif eye == "right" and sbs.eyes.get("right") is not None:
        bgr = _frame_to_bgr(sbs.eyes["right"])
    elif sbs.eyes.get("left") is not None:
        bgr = _frame_to_bgr(sbs.eyes["left"])
    else:
        return None

    bgr = apply_eye_mount_rotate(bgr, _normalize_rotate(eye_rotate))
    h, w = bgr.shape[:2]
    if max_width > 0 and w > max_width:
        scale = max_width / float(w)
        bgr = cv2.resize(
            bgr,
            (max_width, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return bgr
