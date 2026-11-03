"""
NVMM fd → VIC via NvBufSurfTransform（与 MotionLabs jetson_nvbuf_vic_scale 同思路）。

L3：中心裁切 / 整幅缩放；L5：ROI 裁切 + 缩放到姿态输入尺寸。
"""

from __future__ import annotations

import ctypes
import logging
import threading
from typing import Any

from .nv12_util import nv12_er_planes_to_bgr
from .nvbuf_util import (
    NVBUF_COLOR_FORMAT_NV12_ER,
    NVBUF_LAYOUT_PITCH,
    NVBUF_MAP_READ,
    NVBUF_MAP_READ_WRITE,
    NVBUF_MEM_SURFACE_ARRAY,
    nv12_plane_arrays,
    nvbuf_surface_from_fd,
    nvbuf_surface_map,
    nvbuf_surface_map_egl_image,
    nvbuf_surface_sync_for_cpu,
    nvbuf_surface_sync_for_device,
    nvbuf_surface_unmap,
    nvbuf_surface_unmap_egl_image,
)

logger = logging.getLogger(__name__)

NVBUFSURF_TRANSFORM_CROP_SRC = 1
NVBUFSURF_TRANSFORM_CROP_DST = 1 << 1
NVBUFSURF_TRANSFORM_FILTER = 1 << 2
NVBUFSURF_TRANSFORM_FLIP = 1 << 3
NVBUFSURF_TRANSFORM_LETTERBOX_FLAGS = (
    NVBUFSURF_TRANSFORM_CROP_SRC
    | NVBUFSURF_TRANSFORM_CROP_DST
    | NVBUFSURF_TRANSFORM_FILTER
)
NvBufSurfTransformCompute_VIC = 2
NvBufSurfTransformInter_Bilinear = 1
# nvbufsurftransform.h :: NvBufSurfTransform_Flip
NvBufSurfTransform_None = 0
# 注意：与 core.geometry.eye_mount_rotate 的 CPU 语义相反——
# CPU cw90 ≡ Rotate270，CPU ccw90 ≡ Rotate90（见 SbsNv12VicSplitter）。
NvBufSurfTransform_Rotate90 = 1
NvBufSurfTransform_Rotate180 = 2
NvBufSurfTransform_Rotate270 = 3


def center_crop_rect(
    src_w: int, src_h: int, det_w: int, det_h: int
) -> tuple[int, int, int, int]:
    w = min(int(det_w), int(src_w))
    h = min(int(det_h), int(src_h))
    left = max(0, (int(src_w) - w) // 2)
    top = max(0, (int(src_h) - h) // 2)
    return left, top, w, h


class _NvBufSurfTransformRect(ctypes.Structure):
    _fields_ = [
        ("top", ctypes.c_uint32),
        ("left", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
    ]


class _NvBufSurfaceCreateParams(ctypes.Structure):
    _fields_ = [
        ("gpuId", ctypes.c_uint32),
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("isContiguous", ctypes.c_bool),
        ("_pad", ctypes.c_byte * 3),
        ("colorFormat", ctypes.c_int),
        ("layout", ctypes.c_int),
        ("memType", ctypes.c_int),
    ]


class _NvBufSurfTransformConfigParams(ctypes.Structure):
    _fields_ = [
        ("compute_mode", ctypes.c_int),
        ("gpu_id", ctypes.c_int32),
        ("cuda_stream", ctypes.c_void_p),
    ]


class _NvBufSurfTransformParams(ctypes.Structure):
    _fields_ = [
        ("transform_flag", ctypes.c_uint32),
        ("transform_flip", ctypes.c_int),
        ("transform_filter", ctypes.c_int),
        ("src_rect", ctypes.POINTER(_NvBufSurfTransformRect)),
        ("dst_rect", ctypes.POINTER(_NvBufSurfTransformRect)),
    ]


_nvbuf: ctypes.CDLL | None = None
_xform: ctypes.CDLL | None = None
_vic_op_lock = threading.Lock()


def vic_op_lock() -> threading.Lock:
    """全局 VIC 锁；CPU 旋转等勿与 NvBufSurfTransform 并发。"""
    return _vic_op_lock


def _nvbuf_lib() -> ctypes.CDLL:
    global _nvbuf
    if _nvbuf is None:
        from .nvbuf_util import _load

        _nvbuf = _load()
    return _nvbuf


def _xform_lib() -> ctypes.CDLL:
    global _xform
    if _xform is None:
        _xform = ctypes.CDLL("libnvbufsurftransform.so", mode=ctypes.RTLD_GLOBAL)
        cfg = _NvBufSurfTransformConfigParams()
        cfg.compute_mode = NvBufSurfTransformCompute_VIC
        _xform.NvBufSurfTransformSetSessionParams.argtypes = [
            ctypes.POINTER(_NvBufSurfTransformConfigParams)
        ]
        _xform.NvBufSurfTransformSetSessionParams(ctypes.byref(cfg))
        _xform.NvBufSurfTransform.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(_NvBufSurfTransformParams),
        ]
        _xform.NvBufSurfTransform.restype = ctypes.c_int
    return _xform


def nvbuf_vic_scale_available() -> bool:
    try:
        _xform_lib()
        return True
    except OSError:
        return False


def _cuda_ctx_sync() -> None:
    try:
        from cuda.bindings import driver as cuda

        cuda.cuCtxSynchronize()
    except Exception:
        pass


def nvbuf_vic_copy_fd(
    src_fd: int,
    dst_surf: Any,
    *,
    width: int | None = None,
    height: int | None = None,
) -> None:
    """NVMM 同尺寸拷贝 src fd → dst NvBufSurface（session enc 池，NvBufSurfaceCopy）。"""
    del width, height
    _cuda_ctx_sync()
    src = nvbuf_surface_from_fd(int(src_fd))
    nv = _nvbuf_lib()

    def _go() -> None:
        ret = nv.NvBufSurfaceCopy(src, dst_surf)
        if ret != 0:
            raise RuntimeError(f"NvBufSurfaceCopy ret={ret}")

    with _vic_op_lock:
        _go()


class JetsonNvBufVicScaler:
    """可复用 NVMM 输出池 + VIC NvBufSurfTransform。"""

    def __init__(
        self,
        out_w: int,
        out_h: int,
        *,
        prefill_gray114: bool = False,
        transform_flip: int | None = None,
    ) -> None:
        """
        out_w/out_h：VIC 输出表面尺寸。

        transform_flip：可选 NvBufSurfTransform_Rotate90/270 等；与 CROP 同一次
        NvBufSurfTransform。旋转 90/270 时调用方须把 out 设为互换后的宽高
        （例如 src ROI 1920×1080 → out 1080×1920）。
        """
        self.out_w = int(out_w)
        self.out_h = int(out_h)
        self.crop_left = 0
        self.crop_top = 0
        flip = int(transform_flip) if transform_flip is not None else NvBufSurfTransform_None
        self._transform_flip = flip
        self._src_rect = _NvBufSurfTransformRect()
        self._dst_rect = _NvBufSurfTransformRect()
        nv = _nvbuf_lib()
        params = _NvBufSurfaceCreateParams()
        params.width = self.out_w
        params.height = self.out_h
        params.colorFormat = NVBUF_COLOR_FORMAT_NV12_ER
        params.layout = NVBUF_LAYOUT_PITCH
        params.memType = NVBUF_MEM_SURFACE_ARRAY
        out_ptr = ctypes.c_void_p()
        ret = nv.NvBufSurfaceCreate(ctypes.byref(out_ptr), 1, ctypes.byref(params))
        if ret != 0 or not out_ptr.value:
            raise RuntimeError(f"NvBufSurfaceCreate {out_w}x{out_h} ret={ret}")
        self._out_surf_ptr = out_ptr
        from .nvbuf_util import NvBufSurface

        self._out_surf = ctypes.cast(out_ptr, ctypes.POINTER(NvBufSurface))
        self._out_fd = int(self._out_surf.contents.surfaceList[0].bufferDesc)
        # 自有输出池：VIC 写 + 可选 CPU 灰底填充，必须 READ_WRITE
        nvbuf_surface_map(self._out_surf, 0, -1, NVBUF_MAP_READ_WRITE)
        self._y, self._uv, self._ow, self._oh, self._opitch = nv12_plane_arrays(
            self._out_surf, 0
        )
        if prefill_gray114:
            self._write_gray114_cpu_planes()
            # 灰底仅 CPU 写入；device sync 在 EGL/VIC 路径中按需执行
        tp = _NvBufSurfTransformParams()
        flags = NVBUFSURF_TRANSFORM_LETTERBOX_FLAGS
        if flip != NvBufSurfTransform_None:
            flags |= NVBUFSURF_TRANSFORM_FLIP
        tp.transform_flag = flags
        tp.transform_flip = flip
        tp.transform_filter = NvBufSurfTransformInter_Bilinear
        self._dst_rect.top = 0
        self._dst_rect.left = 0
        self._dst_rect.width = self.out_w
        self._dst_rect.height = self.out_h
        tp.src_rect = ctypes.pointer(self._src_rect)
        tp.dst_rect = ctypes.pointer(self._dst_rect)
        self._params = tp
        self._egl_cuda = None
        self._egl_image_ptr: int | None = None
        self._egl_cuda_ok = False
        # 旋转输出仅走 CPU snapshot；EGL 在 1080×1920 + FLIP 上易触发 Invalid_Params
        if flip == NvBufSurfTransform_None:
            try:
                from .nvbuf_egl_cuda import get_shared_egl_cuda_mapper, nvbuf_egl_cuda_available

                if nvbuf_egl_cuda_available():
                    nvbuf_surface_map_egl_image(self._out_surf, 0)
                    egl_ptr = int(
                        self._out_surf.contents.surfaceList[0].mappedAddr.eglImage
                    )
                    if egl_ptr:
                        self._egl_image_ptr = egl_ptr
                        self._egl_cuda = get_shared_egl_cuda_mapper()
                        self._egl_cuda_ok = True
            except Exception as exc:
                logger.debug("NvBuf EGL-CUDA interop unavailable: %s", exc)

    @property
    def out_fd(self) -> int:
        return self._out_fd

    def _write_gray114_cpu_planes(self) -> None:
        h, pitch = self._oh, self._opitch
        self._y[:h, :pitch].fill(114)
        self._uv[: h // 2, :pitch].fill(128)

    def _transform(self, src: Any, sync_device: bool = False) -> None:
        def _go() -> None:
            ret = _xform_lib().NvBufSurfTransform(
                src, self._out_surf, ctypes.byref(self._params)
            )
            if ret != 0:
                raise RuntimeError(f"NvBufSurfTransform(VIC) ret={ret}")
            if self._egl_cuda_ok or sync_device:
                nvbuf_surface_sync_for_device(self._out_surf, 0, -1)
            else:
                nvbuf_surface_sync_for_cpu(self._out_surf, 0, -1)

        with _vic_op_lock:
            _go()

    def _update_crop_rects(self, src_w: int, src_h: int) -> None:
        left, top, cw, ch = center_crop_rect(src_w, src_h, self.out_w, self.out_h)
        self.crop_left = int(left)
        self.crop_top = int(top)
        self._src_rect.top = int(top)
        self._src_rect.left = int(left)
        self._src_rect.width = int(cw)
        self._src_rect.height = int(ch)

    def _update_stretch_rects(self, src_w: int, src_h: int) -> None:
        self.crop_left = 0
        self.crop_top = 0
        self._src_rect.top = 0
        self._src_rect.left = 0
        self._src_rect.width = int(src_w)
        self._src_rect.height = int(src_h)

    def _update_roi_rects(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self.crop_left = 0
        self.crop_top = 0
        self._src_rect.left = int(x1)
        self._src_rect.top = int(y1)
        self._src_rect.width = int(x2 - x1)
        self._src_rect.height = int(y2 - y1)

    def scale_fd(
        self, src_fd: int, src_w: int | None = None, src_h: int | None = None
    ) -> None:
        """VIC 中心裁切 src NVMM fd → 内部缓冲。"""
        src = nvbuf_surface_from_fd(int(src_fd))
        if src_w is None or src_h is None:
            sl = src.contents.surfaceList[0]
            src_w = int(sl.width)
            src_h = int(sl.height)
        self._update_crop_rects(int(src_w), int(src_h))
        self._transform(src)

    def rescale_fd(
        self, src_fd: int, src_w: int | None = None, src_h: int | None = None
    ) -> None:
        """VIC 整幅缩放 src NVMM fd → 内部缓冲。"""
        src = nvbuf_surface_from_fd(int(src_fd))
        if src_w is None or src_h is None:
            sl = src.contents.surfaceList[0]
            src_w = int(sl.width)
            src_h = int(sl.height)
        self._update_stretch_rects(int(src_w), int(src_h))
        self._transform(src)

    def fill_nv12_gray114(self) -> None:
        """LetterBox 灰边：Y=114, UV=128（Ultralytics padding_value）。"""
        with _vic_op_lock:
            if self._egl_cuda_ok:
                try:
                    nvbuf_surface_unmap_egl_image(self._out_surf, 0)
                except Exception:
                    pass
            nvbuf_surface_sync_for_cpu(self._out_surf, 0, -1)
            self._write_gray114_cpu_planes()
            nvbuf_surface_sync_for_device(self._out_surf, 0, -1)
            if self._egl_cuda_ok:
                nvbuf_surface_map_egl_image(self._out_surf, 0)
                egl_ptr = int(self._out_surf.contents.surfaceList[0].mappedAddr.eglImage)
                if egl_ptr:
                    self._egl_image_ptr = egl_ptr

    def place_fd(
        self,
        src_fd: int,
        src_w: int,
        src_h: int,
        dst_left: int,
        dst_top: int,
    ) -> None:
        """
        将 src 整幅贴到本缓冲的 (dst_left, dst_top)（VIC，letterbox 第二步）。

        必须带 CROP_DST：否则 NvBufSurfTransform 会把 src 拉伸到整块 dst 表面，
        640×360 会被拉成 640×640，TRT 输入纵向拉伸 → 叠图骨架拉长/偏上。
        """
        src = nvbuf_surface_from_fd(int(src_fd))
        sw, sh = int(src_w), int(src_h)
        self._src_rect.top = 0
        self._src_rect.left = 0
        self._src_rect.width = sw
        self._src_rect.height = sh
        self._dst_rect.top = int(dst_top)
        self._dst_rect.left = int(dst_left)
        self._dst_rect.width = sw
        self._dst_rect.height = sh
        self._transform(src, sync_device=True)

    def letterbox_fd(
        self,
        src_fd: int,
        src_w: int,
        src_h: int,
        *,
        dst_left: int,
        dst_top: int,
        content_w: int,
        content_h: int,
    ) -> None:
        """
        一步 Ultralytics LetterBox：整幅 src 缩放到输出缓冲的子矩形（灰边区保留）。

        输出缓冲须已 prefill_gray114；仅写入 (dst_left,dst_top)+content 区域。
        """
        src = nvbuf_surface_from_fd(int(src_fd))
        self._src_rect.top = 0
        self._src_rect.left = 0
        self._src_rect.width = int(src_w)
        self._src_rect.height = int(src_h)
        self._dst_rect.top = int(dst_top)
        self._dst_rect.left = int(dst_left)
        self._dst_rect.width = int(content_w)
        self._dst_rect.height = int(content_h)
        self._transform(src, sync_device=True)

    def letterbox_roi_fd(
        self,
        src_fd: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        *,
        dst_left: int,
        dst_top: int,
        content_w: int,
        content_h: int,
    ) -> None:
        """
        SBS ROI 一步 LetterBox：从 src fd 的子矩形缩放到 640 灰底 content 区。

        用于 3840 SBS 直接裁左/右眼 → 640 TRT，跳过中间 1920 split。
        """
        src = nvbuf_surface_from_fd(int(src_fd))
        self._src_rect.left = int(x1)
        self._src_rect.top = int(y1)
        self._src_rect.width = int(x2 - x1)
        self._src_rect.height = int(y2 - y1)
        self._dst_rect.top = int(dst_top)
        self._dst_rect.left = int(dst_left)
        self._dst_rect.width = int(content_w)
        self._dst_rect.height = int(content_h)
        self._transform(src, sync_device=True)

    def roi_rescale_fd(
        self,
        src_fd: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        src_w: int | None = None,
        src_h: int | None = None,
    ) -> None:
        """VIC ROI 裁切 + 缩放到 out_w×out_h。"""
        src = nvbuf_surface_from_fd(int(src_fd))
        if src_w is None or src_h is None:
            sl = src.contents.surfaceList[0]
            src_w = int(sl.width)
            src_h = int(sl.height)
        self._update_roi_rects(int(x1), int(y1), int(x2), int(y2))
        self._transform(src)

    def snapshot_bgr_cpu(self) -> Any:
        nvbuf_surface_sync_for_cpu(self._out_surf, 0, -1)
        h, w, p = self._oh, self._ow, self._opitch
        y = self._y
        uv = self._uv[: h // 2, :w].reshape(h // 2, w // 2, 2)
        return nv12_er_planes_to_bgr(y, uv, h, w, pitch=p)

    def snapshot_nv12_planes_cpu(self) -> tuple[Any, Any, int, int, int]:
        """VIC 输出 NV12_ER 映射平面（L5 直读，无 BGR 转换）。"""
        nvbuf_surface_sync_for_cpu(self._out_surf, 0, -1)
        return self._y, self._uv, int(self._ow), int(self._oh), int(self._opitch)

    def nv12_pitch_cuda_planes(
        self,
        *,
        strict_egl: bool = True,
    ) -> tuple[Any, dict[str, float]]:
        """
        L4 NV12 engine：NVMM det patch → CUDA pitch Y/UV（TRT 直绑）。

        strict_egl=True 时仅 EGL-CUDA 零拷贝，失败直接报错。
        """
        from .nvbuf_egl_cuda import Nv12CudaPitchPlanes

        profile: dict[str, float] = {}
        if self._egl_cuda_ok and self._egl_cuda is not None and self._egl_image_ptr:
            try:
                nvbuf_surface_sync_for_device(self._out_surf, 0, -1)
                planes = self._egl_cuda.egl_nv12_descriptor(
                    self._egl_image_ptr,
                    self._oh,
                    self._ow,
                    self._opitch,
                )
                profile["page_nvmm_egl_cuda_bind"] = 1.0
                profile["l4_nv12_h2d_ms"] = 0.0
                return planes, profile
            except Exception as exc:
                if strict_egl:
                    raise RuntimeError(
                        f"EGL-CUDA NV12 bind failed ({self._ow}×{self._oh} "
                        f"pitch={self._opitch}): {exc}"
                    ) from exc
                logger.debug("EGL-CUDA NV12 bind failed (%s)", exc)
        if strict_egl:
            raise RuntimeError(
                "EGL-CUDA unavailable for NVMM→CUDA NV12 pitch bind "
                "(check nvbuf_egl_cuda / JetPack EGL)"
            )
        raise RuntimeError("NV12 EGL bind required; CPU H2D fallback disabled")

    def release(self) -> None:
        # EGL-CUDA mapper 为进程单例，不在单个 scaler 上 release
        self._egl_cuda = None
        if getattr(self, "_out_surf", None) is not None:
            if self._egl_cuda_ok:
                try:
                    nvbuf_surface_unmap_egl_image(self._out_surf, 0)
                except Exception:
                    pass
            nvbuf_surface_unmap(self._out_surf, 0, -1)
            self._out_surf = None
        ptr = getattr(self, "_out_surf_ptr", None)
        if ptr is not None and ptr.value:
            _nvbuf_lib().NvBufSurfaceDestroy(ptr)
            self._out_surf_ptr = ctypes.c_void_p()
        self._egl_image_ptr = None
        self._egl_cuda_ok = False
