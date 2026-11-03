"""单路 SBS fd：一次 EGL map → 双眼 letterbox + 可选 preview BGR。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from platform.jetson.letterbox_util import LetterboxGeometry
from platform.jetson.nvbuf_egl_cuda import Nv12CudaPitchPlanes, get_shared_egl_cuda_mapper
from platform.jetson.route2_gpu_preprocess import Route2GpuPreprocessor


@dataclass
class SbsGpuPreviewOutput:
    bgr: np.ndarray
    preview_ms: float


class SbsGpuEglSession:
    """同一 SBS fd 只 map/unmap EGL 一次。"""

    def __init__(self) -> None:
        self._surf: Any = None
        self._desc: Nv12CudaPitchPlanes | None = None
        self._mapped: Any = None
        self._stream_handle: int = 0
        self.full_w: int = 0
        self.full_h: int = 0
        self.egl_bind_ms: float = 0.0

    @classmethod
    def open(
        cls,
        src_fd: int,
        full_w: int,
        full_h: int,
        *,
        stream_handle: int,
    ) -> SbsGpuEglSession:
        sess = cls()
        sess.full_w = int(full_w)
        sess.full_h = int(full_h)
        sess._stream_handle = int(stream_handle)
        sess._surf, sess._desc = Route2GpuPreprocessor._map_eye_fd(
            int(src_fd), sess.full_w, sess.full_h
        )
        mapper = get_shared_egl_cuda_mapper()
        t0 = time.perf_counter()
        sess._mapped = mapper.map_nv12_planes(sess._desc, sess._stream_handle)
        sess.egl_bind_ms = (time.perf_counter() - t0) * 1000.0
        return sess

    def close(self) -> None:
        if self._desc is not None and self._mapped is not None:
            get_shared_egl_cuda_mapper().unmap_nv12_planes(
                self._desc, self._stream_handle
            )
        if self._surf is not None:
            Route2GpuPreprocessor._unmap_eye(self._surf)
        self._mapped = None
        self._desc = None
        self._surf = None

    def __enter__(self) -> SbsGpuEglSession:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def letterbox_crop_to_trt(
        self,
        trt: Any,
        gpu: Any,
        *,
        crop_x: int,
        crop_y: int,
        crop_w: int,
        crop_h: int,
        stream_handle: int | None = None,
        dst_ptr: int | None = None,
    ) -> tuple[LetterboxGeometry, float]:
        from platform.jetson.cuda.nv12_rgb import nv12_er_crop_letterbox_to_nchw_rgb

        if self._mapped is None:
            raise RuntimeError("SbsGpuEglSession 未 open")
        stream = int(stream_handle if stream_handle is not None else self._stream_handle)
        out_ptr = int(dst_ptr if dst_ptr is not None else trt.input_device_ptr)
        cw, ch = int(crop_w), int(crop_h)
        geo = gpu.geometry_for(cw, ch)
        t0 = time.perf_counter()
        nv12_er_crop_letterbox_to_nchw_rgb(
            self._mapped.y_ptr,
            self._mapped.uv_ptr,
            out_ptr,
            self.full_w,
            self.full_h,
            int(self._mapped.pitch),
            int(crop_x),
            int(crop_y),
            cw,
            ch,
            gpu.det_w,
            gpu.det_h,
            geo.unpad_w,
            geo.unpad_h,
            geo.pad_left,
            geo.pad_top,
            stream,
        )
        lb_ms = (time.perf_counter() - t0) * 1000.0
        gpu._last_geometry = geo
        return geo, lb_ms

    def letterbox_stereo_crop_to_trt(
        self,
        trt: Any,
        gpu: Any,
        *,
        eye_w: int,
        crop_y: int = 0,
        crop_h: int | None = None,
        stream_handle: int | None = None,
        dst_ptr0: int | None = None,
        dst_ptr1: int | None = None,
    ) -> tuple[LetterboxGeometry, float]:
        from platform.jetson.cuda.nv12_rgb import (
            nv12_er_stereo_sbs_crop_letterbox_to_nchw_rgb,
        )

        if self._mapped is None:
            raise RuntimeError("SbsGpuEglSession 未 open")
        stream = int(stream_handle if stream_handle is not None else self._stream_handle)
        ch = int(crop_h if crop_h is not None else self.full_h)
        ew = int(eye_w)
        geo = gpu.geometry_for(ew, ch)
        out0 = int(dst_ptr0 if dst_ptr0 is not None else trt.input_device_ptr_at(0))
        out1 = int(dst_ptr1 if dst_ptr1 is not None else trt.input_device_ptr_at(1))
        t0 = time.perf_counter()
        nv12_er_stereo_sbs_crop_letterbox_to_nchw_rgb(
            self._mapped.y_ptr,
            self._mapped.uv_ptr,
            out0,
            out1,
            self.full_w,
            self.full_h,
            int(self._mapped.pitch),
            ew,
            int(crop_y),
            ch,
            gpu.det_w,
            gpu.det_h,
            geo.unpad_w,
            geo.unpad_h,
            geo.pad_left,
            geo.pad_top,
            stream,
        )
        lb_ms = (time.perf_counter() - t0) * 1000.0
        gpu._last_geometry = geo
        return geo, lb_ms

    def preview_left_bgr(
        self,
        preview: Any,
        *,
        eye_w: int,
    ) -> SbsGpuPreviewOutput:
        if self._mapped is None:
            raise RuntimeError("SbsGpuEglSession 未 open")
        bgr, preview_ms = preview.fill_from_mapped(
            self._mapped.y_ptr,
            self._mapped.uv_ptr,
            int(self._mapped.pitch),
            self.full_w,
            self.full_h,
            eye_w=eye_w,
            stream_handle=self._stream_handle,
        )
        return SbsGpuPreviewOutput(bgr=bgr, preview_ms=preview_ms)

    def letterbox_crop_to_vitpose_trt(
        self,
        trt: Any,
        *,
        crop_x: int,
        crop_y: int,
        crop_w: int,
        crop_h: int,
        geo: Any,
        stream_handle: int | None = None,
        dst_ptr: int | None = None,
    ) -> float:
        from core.offline.two_stage.vitpose_input import VITPOSE_INPUT_H, VITPOSE_INPUT_W
        from platform.jetson.cuda.nv12_rgb import (
            nv12_er_crop_letterbox_to_nchw_rgb_imagenet,
        )

        if self._mapped is None:
            raise RuntimeError("SbsGpuEglSession 未 open")
        stream = int(stream_handle if stream_handle is not None else self._stream_handle)
        out_ptr = int(dst_ptr if dst_ptr is not None else trt.input_device_ptr)
        t0 = time.perf_counter()
        nv12_er_crop_letterbox_to_nchw_rgb_imagenet(
            self._mapped.y_ptr,
            self._mapped.uv_ptr,
            out_ptr,
            self.full_w,
            self.full_h,
            int(self._mapped.pitch),
            int(crop_x),
            int(crop_y),
            int(crop_w),
            int(crop_h),
            VITPOSE_INPUT_W,
            VITPOSE_INPUT_H,
            int(geo.content_w),
            int(geo.content_h),
            int(geo.pad_left),
            int(geo.pad_top),
            stream,
        )
        return (time.perf_counter() - t0) * 1000.0

    def letterbox_rectify_crop_to_trt(
        self,
        trt: Any,
        gpu: Any,
        *,
        eye_crop_left: int,
        rect_w: int,
        rect_h: int,
        map_x_ptr: int,
        map_y_ptr: int,
        stream_handle: int | None = None,
        dst_ptr: int | None = None,
    ) -> tuple[LetterboxGeometry, float]:
        from platform.jetson.cuda.nv12_rgb import nv12_er_remap_crop_letterbox_to_nchw_rgb

        if self._mapped is None:
            raise RuntimeError("SbsGpuEglSession 未 open")
        stream = int(stream_handle if stream_handle is not None else self._stream_handle)
        out_ptr = int(dst_ptr if dst_ptr is not None else trt.input_device_ptr)
        rw, rh = int(rect_w), int(rect_h)
        geo = gpu.geometry_for(rw, rh)
        t0 = time.perf_counter()
        nv12_er_remap_crop_letterbox_to_nchw_rgb(
            self._mapped.y_ptr,
            self._mapped.uv_ptr,
            int(map_x_ptr),
            int(map_y_ptr),
            out_ptr,
            self.full_w,
            self.full_h,
            int(self._mapped.pitch),
            int(eye_crop_left),
            rw,
            rh,
            gpu.det_w,
            gpu.det_h,
            geo.unpad_w,
            geo.unpad_h,
            geo.pad_left,
            geo.pad_top,
            stream,
        )
        lb_ms = (time.perf_counter() - t0) * 1000.0
        gpu._last_geometry = geo
        return geo, lb_ms

    def letterbox_rectify_vitpose_to_trt(
        self,
        trt: Any,
        *,
        eye_crop_left: int,
        rect_w: int,
        rect_h: int,
        map_x_ptr: int,
        map_y_ptr: int,
        bbox_left: int,
        bbox_top: int,
        bbox_w: int,
        bbox_h: int,
        geo: Any,
        stream_handle: int | None = None,
        dst_ptr: int | None = None,
    ) -> float:
        from core.offline.two_stage.vitpose_input import VITPOSE_INPUT_H, VITPOSE_INPUT_W
        from platform.jetson.cuda.nv12_rgb import (
            nv12_er_remap_bbox_letterbox_to_nchw_rgb_imagenet,
        )

        if self._mapped is None:
            raise RuntimeError("SbsGpuEglSession 未 open")
        stream = int(stream_handle if stream_handle is not None else self._stream_handle)
        out_ptr = int(dst_ptr if dst_ptr is not None else trt.input_device_ptr)
        t0 = time.perf_counter()
        nv12_er_remap_bbox_letterbox_to_nchw_rgb_imagenet(
            self._mapped.y_ptr,
            self._mapped.uv_ptr,
            int(map_x_ptr),
            int(map_y_ptr),
            out_ptr,
            self.full_w,
            self.full_h,
            int(self._mapped.pitch),
            int(eye_crop_left),
            int(rect_w),
            int(rect_h),
            int(bbox_left),
            int(bbox_top),
            int(bbox_w),
            int(bbox_h),
            VITPOSE_INPUT_W,
            VITPOSE_INPUT_H,
            int(geo.content_w),
            int(geo.content_h),
            int(geo.pad_left),
            int(geo.pad_top),
            stream,
        )
        return (time.perf_counter() - t0) * 1000.0
