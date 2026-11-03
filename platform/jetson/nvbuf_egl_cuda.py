"""
NVMM NvBufSurface → EGLImage → CUDA pitch 平面（零拷贝 TRT nv12_y/nv12_uv 绑定）。

无 PyTorch：libcudart primary context + Driver API EGL interop。

Tegra 路径（Jetson）：Register → GetMappedEglFrame，**不要** cuGraphicsMapResources
（桌面 CUDA 用法在 Orin 上会返回 999 UNKNOWN）。
参考 jetson_multimedia_api NvCudaProc.cpp / DeepStream nvinfer。
"""

from __future__ import annotations

import ctypes
import logging
import threading
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_CU_GRAPHICS_REGISTER_FLAGS_NONE = 0
_egl_cuda_lock = threading.Lock()
_shared_mapper: "NvBufEglCudaMapper | None" = None


@dataclass(frozen=True)
class Nv12CudaPitchPlanes:
    """NV12 ER pitch-linear CUDA 平面（map 前 y/uv 可为 0，仅 egl_image_ptr 有效）。"""

    y_ptr: int
    uv_ptr: int
    width: int
    height: int
    pitch: int
    egl_image_ptr: int = 0


def nvbuf_egl_cuda_available() -> bool:
    try:
        from cuda.bindings import driver as cu  # noqa: F401

        from .nvbuf_util import nvbuf_available

        return nvbuf_available()
    except Exception:
        return False


def get_shared_egl_cuda_mapper() -> NvBufEglCudaMapper:
    global _shared_mapper
    if _shared_mapper is None:
        _shared_mapper = NvBufEglCudaMapper()
    return _shared_mapper


def release_shared_egl_cuda_mapper() -> None:
    global _shared_mapper
    with _egl_cuda_lock:
        if _shared_mapper is not None:
            _shared_mapper.release()
            _shared_mapper = None


class NvBufEglCudaMapper:
    """EGLImage → CUgraphicsResource；Tegra 上 Register 后直接 GetMappedEglFrame。"""

    def __init__(self) -> None:
        import ctypes as ct

        from cuda.bindings import driver as cu

        self._cu = cu
        self._cudart = ct.CDLL("libcudart.so.12")
        self._cudart.cudaSetDevice(0)
        self._cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self._cudart.cudaStreamSynchronize.restype = ct.c_int
        # NVIDIA sample 在 EGL 注册前调用 cudaFree(0) 以初始化 runtime context
        self._cudart.cudaFree(ct.c_void_p(0))
        cu.cuInit(0)
        err, self._dev = cu.cuDeviceGet(0)
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuDeviceGet failed: {err}")
        err, cur = cu.cuCtxGetCurrent()
        if err == cu.CUresult.CUDA_SUCCESS and cur:
            self._ctx = cur
            self._primary_ctx_acquired = False
        else:
            err, self._ctx = cu.cuDevicePrimaryCtxRetain(self._dev)
            if err != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuDevicePrimaryCtxRetain failed: {err}")
            err, = cu.cuCtxSetCurrent(self._ctx)
            if err != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"cuCtxSetCurrent failed: {err}")
            self._primary_ctx_acquired = True
        self._resources: dict[int, Any] = {}

    def _set_context(self) -> None:
        err, = self._cu.cuCtxSetCurrent(self._ctx)
        if err != self._cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuCtxSetCurrent failed: {err}")

    def _register(self, egl_image_ptr: int) -> Any:
        cu = self._cu
        self._set_context()
        key = int(egl_image_ptr)
        if key in self._resources:
            return self._resources[key]
        err, res = cu.cuGraphicsEGLRegisterImage(
            key, _CU_GRAPHICS_REGISTER_FLAGS_NONE
        )
        if err != cu.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"cuGraphicsEGLRegisterImage failed: {err}")
        self._resources[key] = res
        return res

    def egl_nv12_descriptor(
        self,
        egl_image_ptr: int,
        height: int,
        width: int,
        pitch: int,
    ) -> Nv12CudaPitchPlanes:
        """仅登记 EGLImage；CUDA 指针在 map_nv12_planes 中获取。"""
        return Nv12CudaPitchPlanes(
            y_ptr=0,
            uv_ptr=0,
            width=int(width),
            height=int(height),
            pitch=int(pitch),
            egl_image_ptr=int(egl_image_ptr),
        )

    def nv12_pitch_planes(
        self,
        egl_image_ptr: int,
        height: int,
        width: int,
        pitch: int,
        *,
        stream_handle: int = 0,
    ) -> Nv12CudaPitchPlanes:
        desc = self.egl_nv12_descriptor(egl_image_ptr, height, width, pitch)
        return self.map_nv12_planes(desc, stream_handle)

    def map_nv12_planes(
        self,
        desc: Nv12CudaPitchPlanes,
        stream_handle: int,
    ) -> Nv12CudaPitchPlanes:
        if not desc.egl_image_ptr:
            raise ValueError("Nv12CudaPitchPlanes 缺少 egl_image_ptr")
        with _egl_cuda_lock:
            cu = self._cu
            self._set_context()
            key = int(desc.egl_image_ptr)
            res = self._register(key)
            err, egl_frame = cu.cuGraphicsResourceGetMappedEglFrame(res, 0, 0)
            if err != cu.CUresult.CUDA_SUCCESS:
                raise RuntimeError(
                    f"cuGraphicsResourceGetMappedEglFrame failed: {err}"
                )

            y_ptr = int(egl_frame.frame.pPitch[0])
            uv_ptr = int(egl_frame.frame.pPitch[1])
            if y_ptr == 0 or uv_ptr == 0:
                raise RuntimeError("EGL NV12 pitch pointers are null")

            return Nv12CudaPitchPlanes(
                y_ptr=y_ptr,
                uv_ptr=uv_ptr,
                width=desc.width,
                height=desc.height,
                pitch=desc.pitch,
                egl_image_ptr=key,
            )

    def unmap_nv12_planes(
        self,
        desc: Nv12CudaPitchPlanes,
        stream_handle: int,
    ) -> None:
        """Tegra EGL 无 Map/Unmap；等待 CUDA stream 完成即可。"""
        sh = int(stream_handle) if stream_handle else 0
        with _egl_cuda_lock:
            self._set_context()
            if sh:
                err = self._cudart.cudaStreamSynchronize(ctypes.c_void_p(sh))
                if err != 0:
                    logger.warning("cudaStreamSynchronize failed: %s", err)
            else:
                err, = self._cu.cuCtxSynchronize()
                if err != self._cu.CUresult.CUDA_SUCCESS:
                    logger.warning("cuCtxSynchronize failed: %s", err)

    def release(self) -> None:
        cu = self._cu
        with _egl_cuda_lock:
            for key, res in list(self._resources.items()):
                try:
                    cu.cuGraphicsUnregisterResource(res)
                except Exception as exc:
                    logger.debug("cuGraphicsUnregisterResource(%s): %s", key, exc)
            self._resources.clear()
            if getattr(self, "_primary_ctx_acquired", False):
                try:
                    cu.cuDevicePrimaryCtxRelease(self._dev)
                except Exception as exc:
                    logger.debug("cuDevicePrimaryCtxRelease: %s", exc)
