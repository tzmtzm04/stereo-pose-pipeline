"""display tee 左眼 NV12 fd → EGL/CUDA BGR（无 CPU map / 无 CPU 色彩转换）。"""

from __future__ import annotations

import time

import cv2
import numpy as np

from platform.jetson.cuda.nv12_rgb import (
    bgr_hwc_rotate90,
    gpu_bgr_rotate90_available,
    gpu_preview_bgr_cuda_available,
    nv12_er_crop_to_bgr_hwc,
)
from platform.jetson.cuda_runtime import CudaBuffer, CudaStream, memcpy_device_to_host_async, sync_stream
from platform.jetson.nv12_frame import Nv12Frame
from core.geometry.eye_mount_rotate import EyeMountRotate, parse_eye_mount_rotate, rotated_eye_hw


def _require_contiguous_dst(dst: np.ndarray, shape: tuple[int, int, int]) -> None:
    h, w, c = shape
    if dst.shape != (h, w, c) or dst.dtype != np.uint8:
        raise ValueError(f"dst shape/dtype 期望 {(h, w, c)} uint8，收到 {dst.shape} {dst.dtype}")
    if not dst.flags["C_CONTIGUOUS"]:
        raise ValueError("dst 须 C_CONTIGUOUS")


class GpuDisplayPreview:
    """display 专用 EGL map + CUDA NV12→BGR；与 infer EGL session 独立。"""

    def __init__(self) -> None:
        self._stream = CudaStream()
        self._bgr_dev: CudaBuffer | None = None
        self._bgr_rot_dev: CudaBuffer | None = None
        self._bgr_hosts: list[np.ndarray | None] = [None, None]
        self._host_idx = 0
        self._dev_w = 0
        self._dev_h = 0
        self._rot_w = 0
        self._rot_h = 0

    @property
    def stream_handle(self) -> int:
        return self._stream.handle

    def _ensure_device(self, w: int, h: int) -> None:
        w, h = int(w), int(h)
        if self._bgr_dev is not None and self._dev_w == w and self._dev_h == h:
            return
        if self._bgr_dev is not None:
            self._bgr_dev.free()
        nbytes = w * h * 3
        self._bgr_dev = CudaBuffer(nbytes)
        self._dev_w, self._dev_h = w, h

    def _ensure_rot_device(self, w: int, h: int) -> None:
        w, h = int(w), int(h)
        if self._bgr_rot_dev is not None and self._rot_w == w and self._rot_h == h:
            return
        if self._bgr_rot_dev is not None:
            self._bgr_rot_dev.free()
        self._bgr_rot_dev = CudaBuffer(w * h * 3)
        self._rot_w, self._rot_h = w, h

    def _d2h_from(self, src_dev: CudaBuffer, dst: np.ndarray, w: int, h: int) -> None:
        nbytes = int(w) * int(h) * 3
        stream = self.stream_handle
        memcpy_device_to_host_async(dst, src_dev.ptr, nbytes, stream)
        sync_stream(stream)

    def _next_host_view(self, w: int, h: int) -> np.ndarray:
        """内部 ping-pong host 缓冲，避免 D2H 后再 copy。"""
        self._host_idx ^= 1
        buf = self._bgr_hosts[self._host_idx]
        if buf is None or buf.shape != (h, w, 3):
            buf = np.empty((h, w, 3), dtype=np.uint8)
            self._bgr_hosts[self._host_idx] = buf
        return buf

    def _d2h_bgr(self, dst: np.ndarray, w: int, h: int) -> None:
        assert self._bgr_dev is not None
        self._d2h_from(self._bgr_dev, dst, w, h)

    def fill_into(
        self,
        dst: np.ndarray,
        fd: int,
        width: int,
        height: int,
    ) -> float:
        """单眼 NV12 fd → CUDA BGR，D2H 直写 dst。"""
        from platform.jetson.nvbuf_egl_cuda import get_shared_egl_cuda_mapper
        from platform.jetson.route2_gpu_preprocess import Route2GpuPreprocessor

        w, h = int(width), int(height)
        _require_contiguous_dst(dst, (h, w, 3))
        stream = self.stream_handle
        surf, desc = Route2GpuPreprocessor._map_eye_fd(int(fd), w, h)
        mapper = get_shared_egl_cuda_mapper()
        t0 = time.perf_counter()
        try:
            mapped = mapper.map_nv12_planes(desc, stream)
            self._convert_planes_into(
                dst,
                int(mapped.y_ptr),
                int(mapped.uv_ptr),
                int(mapped.pitch),
                w,
                h,
            )
            mapper.unmap_nv12_planes(desc, stream)
        finally:
            Route2GpuPreprocessor._unmap_eye(surf)
        return (time.perf_counter() - t0) * 1000.0

    def fill_sbs_into(
        self,
        dst: np.ndarray,
        fd: int,
        full_w: int,
        full_h: int,
        *,
        eye_w: int | None = None,
        crop_x: int = 0,
        eye_rotate: EyeMountRotate | str = "none",
    ) -> float:
        """全幅 SBS fd → CUDA 裁单眼 → 可选 GPU 旋转 → D2H 直写 dst。"""
        from platform.jetson.nvbuf_egl_cuda import get_shared_egl_cuda_mapper
        from platform.jetson.route2_gpu_preprocess import Route2GpuPreprocessor

        rot = parse_eye_mount_rotate(
            eye_rotate if isinstance(eye_rotate, str) else str(eye_rotate)
        )
        fw, fh = int(full_w), int(full_h)
        ew = int(eye_w if eye_w is not None else fw // 2)
        cx = int(crop_x)
        out_h, out_w = rotated_eye_hw(fh, ew, rot)
        _require_contiguous_dst(dst, (out_h, out_w, 3))
        stream = self.stream_handle
        surf, desc = Route2GpuPreprocessor._map_eye_fd(int(fd), fw, fh)
        mapper = get_shared_egl_cuda_mapper()
        t0 = time.perf_counter()
        try:
            mapped = mapper.map_nv12_planes(desc, stream)
            self._convert_planes_crop_into(
                dst,
                int(mapped.y_ptr),
                int(mapped.uv_ptr),
                int(mapped.pitch),
                fw,
                fh,
                ew,
                crop_x=cx,
                eye_rotate=rot,
            )
            mapper.unmap_nv12_planes(desc, stream)
        finally:
            Route2GpuPreprocessor._unmap_eye(surf)
        return (time.perf_counter() - t0) * 1000.0

    def fill_from_fd(self, fd: int, width: int, height: int) -> tuple[np.ndarray, float]:
        w, h = int(width), int(height)
        host = self._next_host_view(w, h)
        preview_ms = self.fill_into(host, fd, w, h)
        return host, preview_ms

    def fill_from_sbs_fd(
        self,
        fd: int,
        full_w: int,
        full_h: int,
        *,
        eye_w: int | None = None,
        crop_x: int = 0,
        eye_rotate: EyeMountRotate | str = "none",
    ) -> tuple[np.ndarray, float]:
        fw, fh = int(full_w), int(full_h)
        ew = int(eye_w if eye_w is not None else fw // 2)
        rot = parse_eye_mount_rotate(
            eye_rotate if isinstance(eye_rotate, str) else str(eye_rotate)
        )
        out_h, out_w = rotated_eye_hw(fh, ew, rot)
        host = self._next_host_view(out_w, out_h)
        preview_ms = self.fill_sbs_into(
            host,
            fd,
            fw,
            fh,
            eye_w=ew,
            crop_x=int(crop_x),
            eye_rotate=rot,
        )
        return host, preview_ms

    def _convert_planes_crop_into(
        self,
        dst: np.ndarray,
        y_ptr: int,
        uv_ptr: int,
        pitch: int,
        full_w: int,
        full_h: int,
        eye_w: int,
        *,
        crop_x: int = 0,
        eye_rotate: EyeMountRotate = "none",
    ) -> None:
        ew, fh = int(eye_w), int(full_h)
        cx = int(crop_x)
        self._ensure_device(ew, fh)
        assert self._bgr_dev is not None
        stream = self.stream_handle
        nv12_er_crop_to_bgr_hwc(
            y_ptr,
            uv_ptr,
            self._bgr_dev.ptr,
            int(full_w),
            fh,
            int(pitch),
            cx,
            0,
            ew,
            fh,
            stream,
        )
        if eye_rotate == "none":
            self._d2h_bgr(dst, ew, fh)
            return
        if not gpu_bgr_rotate90_available():
            raise RuntimeError(
                "GPU rotate90 不可用，请 bash scripts/route2/build_cuda_nv12_rgb.sh"
            )
        out_h, out_w = rotated_eye_hw(fh, ew, eye_rotate)
        self._ensure_rot_device(out_w, out_h)
        assert self._bgr_rot_dev is not None
        bgr_hwc_rotate90(
            self._bgr_dev.ptr,
            self._bgr_rot_dev.ptr,
            fh,
            ew,
            clockwise=(eye_rotate == "cw90"),
            stream_handle=stream,
        )
        self._d2h_from(self._bgr_rot_dev, dst, out_w, out_h)

    def _convert_planes_into(
        self,
        dst: np.ndarray,
        y_ptr: int,
        uv_ptr: int,
        pitch: int,
        width: int,
        height: int,
    ) -> None:
        w, h = int(width), int(height)
        self._ensure_device(w, h)
        assert self._bgr_dev is not None
        stream = self.stream_handle
        nv12_er_crop_to_bgr_hwc(
            y_ptr,
            uv_ptr,
            self._bgr_dev.ptr,
            w,
            h,
            int(pitch),
            0,
            0,
            w,
            h,
            stream,
        )
        self._d2h_bgr(dst, w, h)


_preview: GpuDisplayPreview | None = None


def get_gpu_display_preview() -> GpuDisplayPreview:
    global _preview
    if _preview is None:
        _preview = GpuDisplayPreview()
    return _preview


def gpu_display_preview_available() -> bool:
    return gpu_preview_bgr_cuda_available()


def _is_sbs_full_frame(frame: Nv12Frame) -> bool:
    """宽屏 SBS（如 3840×1080）vs 已裁好的单眼（1920×1080）。"""
    return int(frame.width) > int(frame.height) * 2


def preview_eye_shape(frame: Nv12Frame) -> tuple[int, int]:
    """display BGR 输出 (height, width)。"""
    if _is_sbs_full_frame(frame):
        return int(frame.height), int(frame.width) // 2
    return int(frame.height), int(frame.width)


def nv12_display_frame_to_preview_bgr_into(
    frame: Nv12Frame,
    dst: np.ndarray,
    *,
    max_width: int = 0,
    gpu_preview: GpuDisplayPreview | None = None,
    display_eye: str = "left",
) -> float:
    """display NV12 → BGR，D2H 直写 dst（P2 零中间拷贝）。"""
    if frame.dmabuf_fd is None:
        raise RuntimeError("display GPU preview 需要 dmabuf_fd")
    from core.display.display_eye import normalize_display_eye, sbs_crop_x_for_eye

    eye = normalize_display_eye(display_eye)
    preview = gpu_preview or get_gpu_display_preview()
    if _is_sbs_full_frame(frame):
        preview_ms = preview.fill_sbs_into(
            dst,
            int(frame.dmabuf_fd),
            int(frame.width),
            int(frame.height),
            crop_x=sbs_crop_x_for_eye(int(frame.width), eye),
        )
    else:
        preview_ms = preview.fill_into(
            dst,
            int(frame.dmabuf_fd),
            int(frame.width),
            int(frame.height),
        )
    if max_width > 0 and dst.shape[1] > max_width:
        h, w = dst.shape[:2]
        scale = max_width / float(w)
        t0 = time.perf_counter()
        resized = cv2.resize(
            dst,
            (max_width, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        np.copyto(dst[: resized.shape[0], : resized.shape[1]], resized)
        preview_ms += (time.perf_counter() - t0) * 1000.0
    return preview_ms


def nv12_display_frame_to_preview_bgr(
    frame: Nv12Frame,
    *,
    max_width: int = 0,
    gpu_preview: GpuDisplayPreview | None = None,
    display_eye: str = "left",
) -> tuple[np.ndarray, float]:
    """display NV12 → BGR；全幅 SBS 时 CUDA 裁单眼，单眼 fd 时直转。"""
    if frame.dmabuf_fd is None:
        raise RuntimeError("display GPU preview 需要 dmabuf_fd")
    from core.display.display_eye import normalize_display_eye, sbs_crop_x_for_eye

    eye = normalize_display_eye(display_eye)
    preview = gpu_preview or get_gpu_display_preview()
    if _is_sbs_full_frame(frame):
        bgr, preview_ms = preview.fill_from_sbs_fd(
            int(frame.dmabuf_fd),
            int(frame.width),
            int(frame.height),
            crop_x=sbs_crop_x_for_eye(int(frame.width), eye),
        )
    else:
        bgr, preview_ms = preview.fill_from_fd(
            int(frame.dmabuf_fd),
            int(frame.width),
            int(frame.height),
        )
    if max_width > 0 and bgr.shape[1] > max_width:
        h, w = bgr.shape[:2]
        scale = max_width / float(w)
        t0 = time.perf_counter()
        bgr = cv2.resize(
            bgr,
            (max_width, max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        preview_ms += (time.perf_counter() - t0) * 1000.0
    return bgr, preview_ms
