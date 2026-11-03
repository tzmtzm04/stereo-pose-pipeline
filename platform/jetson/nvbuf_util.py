"""ctypes bindings for libnvbufsurface (Jetson NVMM / DMA-BUF)."""

from __future__ import annotations

import ctypes
from typing import Any

NVBUF_MAX_PLANES = 4

NVBUF_COLOR_FORMAT_NV12 = 6
NVBUF_COLOR_FORMAT_NV12_ER = 7
NVBUF_MEM_SURFACE_ARRAY = 4
NVBUF_LAYOUT_PITCH = 0
NVBUF_MAP_READ = 0
NVBUF_MAP_READ_WRITE = 2

VIC_NV12_PITCH_ALIGN = 64

_lib: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL("libnvbufsurface.so", mode=ctypes.RTLD_GLOBAL)
    return _lib


class NvBufSurfaceMappedAddr(ctypes.Structure):
    _fields_ = [
        ("addr", ctypes.c_void_p * NVBUF_MAX_PLANES),
        ("eglImage", ctypes.c_void_p),
        ("_reserved", ctypes.c_void_p * 4),
    ]


class NvBufSurfaceParams(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_uint32),
        ("height", ctypes.c_uint32),
        ("pitch", ctypes.c_uint32),
        ("colorFormat", ctypes.c_int),
        ("layout", ctypes.c_int),
        ("_pad0", ctypes.c_uint32),
        ("bufferDesc", ctypes.c_uint64),
        ("dataSize", ctypes.c_uint32),
        ("_pad1", ctypes.c_uint32),
        ("dataPtr", ctypes.c_void_p),
        ("_planeParams", ctypes.c_byte * 232),
        ("mappedAddr", NvBufSurfaceMappedAddr),
        ("paramex", ctypes.c_void_p),
        ("_reserved", ctypes.c_void_p * 3),
    ]


class NvBufSurface(ctypes.Structure):
    _fields_ = [
        ("gpuId", ctypes.c_uint32),
        ("batchSize", ctypes.c_uint32),
        ("numFilled", ctypes.c_uint32),
        ("isContiguous", ctypes.c_bool),
        ("_pad", ctypes.c_byte * 3),
        ("memType", ctypes.c_int),
        ("_pad2", ctypes.c_uint32),
        ("surfaceList", ctypes.POINTER(NvBufSurfaceParams)),
        ("_reserved", ctypes.c_void_p * 4),
    ]


def nvbuf_pitch_vic_compatible(pitch: int) -> bool:
    return int(pitch) > 0 and int(pitch) % VIC_NV12_PITCH_ALIGN == 0


def nvbuf_require_vic_pitch(
    surf: ctypes.POINTER(NvBufSurface), index: int = 0
) -> int:
    p = surf.contents.surfaceList[index]
    layout = int(p.layout)
    pitch = int(p.pitch)
    w, h = int(p.width), int(p.height)
    if layout != NVBUF_LAYOUT_PITCH:
        raise RuntimeError(
            f"NVMM layout={layout} not pitch-linear; VIC requires bl-output=false"
        )
    if not nvbuf_pitch_vic_compatible(pitch):
        raise RuntimeError(
            f"NVMM pitch={pitch} (WxH={w}x{h}) not {VIC_NV12_PITCH_ALIGN}B-aligned"
        )
    return pitch


def nvbuf_available() -> bool:
    try:
        _load()
        return True
    except OSError:
        return False


def nvbuf_surface_from_gst_map(map_data: Any) -> ctypes.POINTER(NvBufSurface):
    return ctypes.cast(map_data, ctypes.POINTER(NvBufSurface))


def nvbuf_surface_from_fd(dmabuf_fd: int) -> ctypes.POINTER(NvBufSurface):
    lib = _load()
    surf = ctypes.c_void_p()
    ret = lib.NvBufSurfaceFromFd(int(dmabuf_fd), ctypes.byref(surf))
    if ret != 0 or not surf.value:
        raise RuntimeError(f"NvBufSurfaceFromFd failed for fd={dmabuf_fd} ret={ret}")
    return ctypes.cast(surf, ctypes.POINTER(NvBufSurface))


def nvbuf_surface_sync_for_cpu(
    surf: ctypes.POINTER(NvBufSurface), index: int = 0, plane: int = -1
) -> None:
    lib = _load()
    ret = lib.NvBufSurfaceSyncForCpu(surf, index, plane)
    if ret != 0:
        raise RuntimeError(f"NvBufSurfaceSyncForCpu failed ret={ret}")


def nvbuf_surface_sync_for_device(
    surf: ctypes.POINTER(NvBufSurface), index: int = 0, plane: int = -1
) -> None:
    lib = _load()
    ret = lib.NvBufSurfaceSyncForDevice(surf, index, plane, 0)
    if ret != 0:
        raise RuntimeError(f"NvBufSurfaceSyncForDevice failed ret={ret}")


def nvbuf_surface_map(
    surf: ctypes.POINTER(NvBufSurface),
    index: int,
    plane: int,
    flags: int,
) -> None:
    lib = _load()
    ret = lib.NvBufSurfaceMap(surf, index, plane, flags)
    if ret != 0:
        raise RuntimeError(f"NvBufSurfaceMap failed ret={ret}")


def nvbuf_surface_unmap(
    surf: ctypes.POINTER(NvBufSurface), index: int = 0, plane: int = -1
) -> None:
    lib = _load()
    ret = lib.NvBufSurfaceUnMap(surf, index, plane)
    if ret != 0:
        raise RuntimeError(f"NvBufSurfaceUnMap failed ret={ret}")


def nvbuf_surface_map_egl_image(
    surf: ctypes.POINTER(NvBufSurface), index: int = 0
) -> None:
    """NVMM SURFACE_ARRAY → EGLImageKHR（供 CUDA-EGL interop 零拷贝读）。"""
    lib = _load()
    ret = lib.NvBufSurfaceMapEglImage(surf, index)
    if ret != 0:
        raise RuntimeError(f"NvBufSurfaceMapEglImage failed ret={ret}")


def nvbuf_surface_unmap_egl_image(
    surf: ctypes.POINTER(NvBufSurface), index: int = 0
) -> None:
    lib = _load()
    ret = lib.NvBufSurfaceUnMapEglImage(surf, index)
    if ret != 0:
        raise RuntimeError(f"NvBufSurfaceUnMapEglImage failed ret={ret}")


def nv12_plane_arrays(
    surf: ctypes.POINTER(NvBufSurface), index: int = 0
) -> tuple[Any, Any, int, int, int]:
    import numpy as np

    p = surf.contents.surfaceList[index]
    w, h, pitch = int(p.width), int(p.height), int(p.pitch)
    y_ptr = p.mappedAddr.addr[0]
    uv_ptr = p.mappedAddr.addr[1]
    if not y_ptr or not uv_ptr:
        raise RuntimeError("NvBufSurface planes not mapped")
    y = np.ctypeslib.as_array(
        ctypes.cast(y_ptr, ctypes.POINTER(ctypes.c_uint8)), shape=(h, pitch)
    )
    uv_h = h // 2
    uv = np.ctypeslib.as_array(
        ctypes.cast(uv_ptr, ctypes.POINTER(ctypes.c_uint8)),
        shape=(uv_h, pitch),
    )
    return y, uv, w, h, pitch
