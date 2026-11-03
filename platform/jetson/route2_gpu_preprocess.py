"""NVMM NV12 eye fd → GPU letterbox + NV12→RGB → TRT（对齐 YOLOs-CPP / Ultralytics）。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from platform.jetson.letterbox_util import LetterboxGeometry, ultralytics_letterbox_geometry
from platform.jetson.nvbuf_egl_cuda import Nv12CudaPitchPlanes, get_shared_egl_cuda_mapper

logger = logging.getLogger(__name__)

DEFAULT_ROUTE2_INPUT = (640, 640)


@dataclass
class GpuLetterboxProfile:
    letterbox_ms: float = 0.0
    egl_bind_ms: float = 0.0

    @property
    def gpu_total_ms(self) -> float:
        return self.letterbox_ms + self.egl_bind_ms


def gpu_letterbox_available() -> bool:
    from platform.jetson.cuda.nv12_rgb import gpu_letterbox_cuda_available

    return gpu_letterbox_cuda_available()


class Route2GpuPreprocessor:
    """1920×1080 NVMM fd → CUDA letterbox 640 + NCHW RGB 直写 TRT input。"""

    def __init__(self, det_size: tuple[int, int] = DEFAULT_ROUTE2_INPUT) -> None:
        if not gpu_letterbox_available():
            raise RuntimeError(
                "GPU letterbox 需要 route2_nv12_rgb.so: bash scripts/route2/build_cuda_nv12_rgb.sh"
            )
        self.det_w, self.det_h = int(det_size[0]), int(det_size[1])
        self._last_geometry: LetterboxGeometry | None = None

    @property
    def last_geometry(self) -> LetterboxGeometry | None:
        return self._last_geometry

    def geometry_for(self, src_w: int, src_h: int) -> LetterboxGeometry:
        return ultralytics_letterbox_geometry(
            int(src_w), int(src_h), self.det_w, self.det_h
        )

    @staticmethod
    def _map_eye_fd(src_fd: int, width: int, height: int) -> tuple[Any, Nv12CudaPitchPlanes]:
        from platform.jetson.nvbuf_util import (
            nvbuf_require_vic_pitch,
            nvbuf_surface_from_fd,
            nvbuf_surface_map_egl_image,
        )

        surf = nvbuf_surface_from_fd(int(src_fd))
        pitch = int(nvbuf_require_vic_pitch(surf, 0))
        nvbuf_surface_map_egl_image(surf, 0)
        egl_ptr = int(surf.contents.surfaceList[0].mappedAddr.eglImage)
        if egl_ptr == 0:
            raise RuntimeError("eye fd EGL map 失败")
        desc = Nv12CudaPitchPlanes(
            y_ptr=0,
            uv_ptr=0,
            width=int(width),
            height=int(height),
            pitch=pitch,
            egl_image_ptr=egl_ptr,
        )
        return surf, desc

    @staticmethod
    def _unmap_eye(surf: Any) -> None:
        from platform.jetson.nvbuf_util import nvbuf_surface_unmap_egl_image

        try:
            nvbuf_surface_unmap_egl_image(surf, 0)
        except Exception:
            pass

    def feed_trt_from_fd(
        self,
        trt: Any,
        src_fd: int,
        src_w: int,
        src_h: int,
        *,
        crop_x: int = 0,
        crop_y: int = 0,
        crop_w: int | None = None,
        crop_h: int | None = None,
    ) -> tuple[LetterboxGeometry, GpuLetterboxProfile]:
        from platform.jetson.cuda.nv12_rgb import (
            nv12_er_crop_letterbox_to_nchw_rgb,
            nv12_er_letterbox_to_nchw_rgb,
        )

        full_w, full_h = int(src_w), int(src_h)
        cw = int(crop_w if crop_w is not None else full_w)
        ch = int(crop_h if crop_h is not None else full_h)
        geo = self.geometry_for(cw, ch)
        prof = GpuLetterboxProfile()
        surf, desc = self._map_eye_fd(src_fd, full_w, full_h)
        mapper = get_shared_egl_cuda_mapper()
        stream = trt.stream.handle
        use_crop = crop_x != 0 or crop_y != 0 or cw != full_w or ch != full_h
        try:
            t0 = time.perf_counter()
            mapped = mapper.map_nv12_planes(desc, stream)
            prof.egl_bind_ms = (time.perf_counter() - t0) * 1000.0
            t1 = time.perf_counter()
            if use_crop:
                nv12_er_crop_letterbox_to_nchw_rgb(
                    mapped.y_ptr,
                    mapped.uv_ptr,
                    trt.input_device_ptr,
                    full_w,
                    full_h,
                    int(mapped.pitch),
                    int(crop_x),
                    int(crop_y),
                    cw,
                    ch,
                    self.det_w,
                    self.det_h,
                    geo.unpad_w,
                    geo.unpad_h,
                    geo.pad_left,
                    geo.pad_top,
                    stream,
                )
            else:
                nv12_er_letterbox_to_nchw_rgb(
                    mapped.y_ptr,
                    mapped.uv_ptr,
                    trt.input_device_ptr,
                    cw,
                    ch,
                    int(mapped.pitch),
                    self.det_w,
                    self.det_h,
                    geo.unpad_w,
                    geo.unpad_h,
                    geo.pad_left,
                    geo.pad_top,
                    stream,
                )
            prof.letterbox_ms = (time.perf_counter() - t1) * 1000.0
            mapper.unmap_nv12_planes(desc, stream)
        finally:
            self._unmap_eye(surf)
        self._last_geometry = geo
        trt.last_preprocess_ms = prof.gpu_total_ms
        return geo, prof
