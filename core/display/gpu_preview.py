"""SBS NV12 fd → GPU 裁左眼 → BGR（display；优先走 SbsGpuEglSession 合并 map）。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from platform.jetson.cuda.nv12_rgb import gpu_preview_bgr_cuda_available, nv12_er_crop_to_bgr_hwc
from platform.jetson.cuda_runtime import CudaBuffer, memcpy_device_to_host_async, sync_stream


class GpuSbsPreview:
    """复用 device/host BGR 缓冲。"""

    def __init__(self) -> None:
        self._bgr_dev: CudaBuffer | None = None
        self._bgr_host: np.ndarray | None = None
        self._dev_w = 0
        self._dev_h = 0

    def _ensure_buffers(self, w: int, h: int) -> None:
        w, h = int(w), int(h)
        if self._bgr_dev is not None and self._dev_w == w and self._dev_h == h:
            return
        if self._bgr_dev is not None:
            self._bgr_dev.free()
        nbytes = w * h * 3
        self._bgr_dev = CudaBuffer(nbytes)
        self._bgr_host = np.empty((h, w, 3), dtype=np.uint8)
        self._dev_w, self._dev_h = w, h

    def fill_from_mapped(
        self,
        y_ptr: int,
        uv_ptr: int,
        pitch: int,
        full_w: int,
        full_h: int,
        *,
        eye_w: int,
        stream_handle: int,
    ) -> tuple[np.ndarray, float]:
        if not gpu_preview_bgr_cuda_available():
            raise RuntimeError("GPU preview kernel 不可用")
        ew, eh = int(eye_w), int(full_h)
        self._ensure_buffers(ew, eh)
        assert self._bgr_dev is not None and self._bgr_host is not None

        t0 = time.perf_counter()
        nv12_er_crop_to_bgr_hwc(
            int(y_ptr),
            int(uv_ptr),
            self._bgr_dev.ptr,
            int(full_w),
            int(full_h),
            int(pitch),
            0,
            0,
            ew,
            eh,
            stream_handle,
        )
        nbytes = ew * eh * 3
        memcpy_device_to_host_async(
            self._bgr_host, self._bgr_dev.ptr, nbytes, stream_handle
        )
        sync_stream(stream_handle)
        preview_ms = (time.perf_counter() - t0) * 1000.0
        return self._bgr_host.copy(), preview_ms

    def preview_from_sbs(
        self,
        sbs: Any,
        *,
        stream_handle: int,
    ) -> tuple[np.ndarray | None, float]:
        """独立 EGL map（无合并 session 时的回退）。"""
        if not gpu_preview_bgr_cuda_available():
            return None, 0.0
        full = getattr(sbs, "full", None)
        if full is None or full.dmabuf_fd is None:
            return None, 0.0

        full_w = int(getattr(sbs, "width", full.width))
        full_h = int(getattr(sbs, "height", full.height))
        eye_w = max(1, full_w // 2)

        from platform.jetson.route2_gpu_preprocess import Route2GpuPreprocessor
        from platform.jetson.nvbuf_egl_cuda import get_shared_egl_cuda_mapper

        surf, desc = Route2GpuPreprocessor._map_eye_fd(
            int(full.dmabuf_fd), full_w, full_h
        )
        mapper = get_shared_egl_cuda_mapper()
        t0 = time.perf_counter()
        try:
            mapped = mapper.map_nv12_planes(desc, stream_handle)
            bgr, _ = self.fill_from_mapped(
                mapped.y_ptr,
                mapped.uv_ptr,
                int(mapped.pitch),
                full_w,
                full_h,
                eye_w=eye_w,
                stream_handle=stream_handle,
            )
            mapper.unmap_nv12_planes(desc, stream_handle)
        finally:
            Route2GpuPreprocessor._unmap_eye(surf)
        preview_ms = (time.perf_counter() - t0) * 1000.0
        return bgr, preview_ms


_preview: GpuSbsPreview | None = None


def get_gpu_sbs_preview() -> GpuSbsPreview:
    global _preview
    if _preview is None:
        _preview = GpuSbsPreview()
    return _preview


def gpu_preview_bgr_from_sbs(
    sbs: Any,
    *,
    stream_handle: int,
) -> tuple[np.ndarray | None, float]:
    return get_gpu_sbs_preview().preview_from_sbs(sbs, stream_handle=stream_handle)
