"""加载 route2_nv12_rgb.so：NV12 ER CUDA pitch → TRT NCHW RGB。"""

from __future__ import annotations

import ctypes
from pathlib import Path

from app.paths import BUILD_ROUTE2, ROOT

_LIB_NAME = "route2_nv12_rgb.so"
_SEARCH = [
    BUILD_ROUTE2 / _LIB_NAME,
    ROOT / "src" / "route2" / _LIB_NAME,
    Path(__file__).resolve().parent / _LIB_NAME,
]

_lib: ctypes.CDLL | None = None


def cuda_nv12_rgb_available() -> bool:
    return _load_lib() is not None


def _load_lib() -> ctypes.CDLL | None:
    global _lib
    if _lib is not None:
        return _lib
    for p in _SEARCH:
        if p.is_file():
            lib = ctypes.CDLL(str(p))
            lib.route2_nv12_er_pitch_to_nchw_rgb.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
            ]
            lib.route2_nv12_er_pitch_to_nchw_rgb.restype = ctypes.c_int
            if hasattr(lib, "route2_nv12_er_letterbox_to_nchw_rgb"):
                lib.route2_nv12_er_letterbox_to_nchw_rgb.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_void_p,
                ]
                lib.route2_nv12_er_letterbox_to_nchw_rgb.restype = ctypes.c_int
            if hasattr(lib, "route2_nv12_er_crop_letterbox_to_nchw_rgb"):
                lib.route2_nv12_er_crop_letterbox_to_nchw_rgb.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_void_p,
                ]
                lib.route2_nv12_er_crop_letterbox_to_nchw_rgb.restype = ctypes.c_int
            if hasattr(lib, "route2_nv12_er_stereo_sbs_crop_letterbox_to_nchw_rgb"):
                lib.route2_nv12_er_stereo_sbs_crop_letterbox_to_nchw_rgb.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_void_p,
                ]
                lib.route2_nv12_er_stereo_sbs_crop_letterbox_to_nchw_rgb.restype = (
                    ctypes.c_int
                )
            if hasattr(lib, "route2_nv12_er_crop_to_bgr_hwc"):
                lib.route2_nv12_er_crop_to_bgr_hwc.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_void_p,
                ]
                lib.route2_nv12_er_crop_to_bgr_hwc.restype = ctypes.c_int
            if hasattr(lib, "route2_bgr_hwc_rotate90"):
                lib.route2_bgr_hwc_rotate90.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_void_p,
                ]
                lib.route2_bgr_hwc_rotate90.restype = ctypes.c_int
            if hasattr(lib, "route2_nv12_er_crop_letterbox_to_nchw_rgb_imagenet"):
                lib.route2_nv12_er_crop_letterbox_to_nchw_rgb_imagenet.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_void_p,
                ]
                lib.route2_nv12_er_crop_letterbox_to_nchw_rgb_imagenet.restype = (
                    ctypes.c_int
                )
            if hasattr(lib, "route2_nv12_er_remap_crop_letterbox_to_nchw_rgb"):
                lib.route2_nv12_er_remap_crop_letterbox_to_nchw_rgb.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_void_p,
                ]
                lib.route2_nv12_er_remap_crop_letterbox_to_nchw_rgb.restype = (
                    ctypes.c_int
                )
            if hasattr(lib, "route2_nv12_er_remap_bbox_letterbox_to_nchw_rgb_imagenet"):
                lib.route2_nv12_er_remap_bbox_letterbox_to_nchw_rgb_imagenet.argtypes = [
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_int,
                    ctypes.c_void_p,
                ]
                lib.route2_nv12_er_remap_bbox_letterbox_to_nchw_rgb_imagenet.restype = (
                    ctypes.c_int
                )
            _lib = lib
            return _lib
    return None


def gpu_letterbox_cuda_available() -> bool:
    lib = _load_lib()
    return lib is not None and hasattr(lib, "route2_nv12_er_letterbox_to_nchw_rgb")


def gpu_crop_letterbox_cuda_available() -> bool:
    lib = _load_lib()
    return lib is not None and hasattr(lib, "route2_nv12_er_crop_letterbox_to_nchw_rgb")


def gpu_stereo_sbs_letterbox_cuda_available() -> bool:
    lib = _load_lib()
    return (
        lib is not None
        and hasattr(lib, "route2_nv12_er_stereo_sbs_crop_letterbox_to_nchw_rgb")
    )


def nv12_er_pitch_to_nchw_rgb(
    y_ptr: int,
    uv_ptr: int,
    dst_ptr: int,
    width: int,
    height: int,
    pitch: int,
    stream_handle: int = 0,
) -> None:
    lib = _load_lib()
    if lib is None:
        raise RuntimeError(
            f"未找到 {_LIB_NAME}，请先运行: bash scripts/route2/build_cuda_nv12_rgb.sh"
        )
    ret = lib.route2_nv12_er_pitch_to_nchw_rgb(
        ctypes.c_void_p(y_ptr),
        ctypes.c_void_p(uv_ptr),
        ctypes.c_void_p(dst_ptr),
        int(width),
        int(height),
        int(pitch),
        ctypes.c_void_p(stream_handle),
    )
    if ret != 0:
        raise RuntimeError(f"route2_nv12_er_pitch_to_nchw_rgb failed code={ret}")


def nv12_er_letterbox_to_nchw_rgb(
    y_ptr: int,
    uv_ptr: int,
    dst_ptr: int,
    src_w: int,
    src_h: int,
    pitch: int,
    dst_w: int,
    dst_h: int,
    unpad_w: int,
    unpad_h: int,
    pad_left: int,
    pad_top: int,
    stream_handle: int = 0,
) -> None:
    lib = _load_lib()
    if lib is None or not hasattr(lib, "route2_nv12_er_letterbox_to_nchw_rgb"):
        raise RuntimeError(
            f"GPU letterbox 未构建，请重新运行: bash scripts/route2/build_cuda_nv12_rgb.sh"
        )
    ret = lib.route2_nv12_er_letterbox_to_nchw_rgb(
        ctypes.c_void_p(y_ptr),
        ctypes.c_void_p(uv_ptr),
        ctypes.c_void_p(dst_ptr),
        int(src_w),
        int(src_h),
        int(pitch),
        int(dst_w),
        int(dst_h),
        int(unpad_w),
        int(unpad_h),
        int(pad_left),
        int(pad_top),
        ctypes.c_void_p(stream_handle),
    )
    if ret != 0:
        raise RuntimeError(f"route2_nv12_er_letterbox_to_nchw_rgb failed code={ret}")


def nv12_er_crop_letterbox_to_nchw_rgb(
    y_ptr: int,
    uv_ptr: int,
    dst_ptr: int,
    full_w: int,
    full_h: int,
    pitch: int,
    crop_left: int,
    crop_top: int,
    crop_w: int,
    crop_h: int,
    dst_w: int,
    dst_h: int,
    unpad_w: int,
    unpad_h: int,
    pad_left: int,
    pad_top: int,
    stream_handle: int = 0,
) -> None:
    lib = _load_lib()
    if lib is None or not hasattr(lib, "route2_nv12_er_crop_letterbox_to_nchw_rgb"):
        raise RuntimeError(
            "GPU crop letterbox 未构建，请重新运行: bash scripts/route2/build_cuda_nv12_rgb.sh"
        )
    ret = lib.route2_nv12_er_crop_letterbox_to_nchw_rgb(
        ctypes.c_void_p(y_ptr),
        ctypes.c_void_p(uv_ptr),
        ctypes.c_void_p(dst_ptr),
        int(full_w),
        int(full_h),
        int(pitch),
        int(crop_left),
        int(crop_top),
        int(crop_w),
        int(crop_h),
        int(dst_w),
        int(dst_h),
        int(unpad_w),
        int(unpad_h),
        int(pad_left),
        int(pad_top),
        ctypes.c_void_p(stream_handle),
    )
    if ret != 0:
        raise RuntimeError(f"route2_nv12_er_crop_letterbox_to_nchw_rgb failed code={ret}")


def nv12_er_stereo_sbs_crop_letterbox_to_nchw_rgb(
    y_ptr: int,
    uv_ptr: int,
    dst_ptr0: int,
    dst_ptr1: int,
    full_w: int,
    full_h: int,
    pitch: int,
    eye_w: int,
    crop_top: int,
    crop_h: int,
    dst_w: int,
    dst_h: int,
    unpad_w: int,
    unpad_h: int,
    pad_left: int,
    pad_top: int,
    stream_handle: int = 0,
) -> None:
    lib = _load_lib()
    if lib is None or not hasattr(
        lib, "route2_nv12_er_stereo_sbs_crop_letterbox_to_nchw_rgb"
    ):
        raise RuntimeError(
            "GPU stereo letterbox 未构建，请重新运行: "
            "bash scripts/route2/build_cuda_nv12_rgb.sh"
        )
    ret = lib.route2_nv12_er_stereo_sbs_crop_letterbox_to_nchw_rgb(
        ctypes.c_void_p(y_ptr),
        ctypes.c_void_p(uv_ptr),
        ctypes.c_void_p(dst_ptr0),
        ctypes.c_void_p(dst_ptr1),
        int(full_w),
        int(full_h),
        int(pitch),
        int(eye_w),
        int(crop_top),
        int(crop_h),
        int(dst_w),
        int(dst_h),
        int(unpad_w),
        int(unpad_h),
        int(pad_left),
        int(pad_top),
        ctypes.c_void_p(stream_handle),
    )
    if ret != 0:
        raise RuntimeError(
            f"route2_nv12_er_stereo_sbs_crop_letterbox_to_nchw_rgb failed code={ret}"
        )


def gpu_preview_bgr_cuda_available() -> bool:
    lib = _load_lib()
    return lib is not None and hasattr(lib, "route2_nv12_er_crop_to_bgr_hwc")


def gpu_vitpose_letterbox_cuda_available() -> bool:
    lib = _load_lib()
    return lib is not None and hasattr(
        lib, "route2_nv12_er_crop_letterbox_to_nchw_rgb_imagenet"
    )


def gpu_rectify_letterbox_cuda_available() -> bool:
    lib = _load_lib()
    return lib is not None and hasattr(
        lib, "route2_nv12_er_remap_crop_letterbox_to_nchw_rgb"
    )


def gpu_rectify_vitpose_cuda_available() -> bool:
    lib = _load_lib()
    return lib is not None and hasattr(
        lib, "route2_nv12_er_remap_bbox_letterbox_to_nchw_rgb_imagenet"
    )


def gpu_rectify_cuda_available() -> bool:
    return bool(
        gpu_rectify_letterbox_cuda_available()
        and gpu_rectify_vitpose_cuda_available()
        and gpu_vitpose_letterbox_cuda_available()
        and gpu_crop_letterbox_cuda_available()
    )


def nv12_er_crop_letterbox_to_nchw_rgb_imagenet(
    y_ptr: int,
    uv_ptr: int,
    dst_ptr: int,
    full_w: int,
    full_h: int,
    pitch: int,
    crop_left: int,
    crop_top: int,
    crop_w: int,
    crop_h: int,
    dst_w: int,
    dst_h: int,
    new_w: int,
    new_h: int,
    pad_left: int,
    pad_top: int,
    stream_handle: int = 0,
) -> None:
    lib = _load_lib()
    if lib is None or not hasattr(
        lib, "route2_nv12_er_crop_letterbox_to_nchw_rgb_imagenet"
    ):
        raise RuntimeError(
            "ViTPose GPU letterbox 未构建，请重新运行: "
            "bash scripts/route2/build_cuda_nv12_rgb.sh"
        )
    ret = lib.route2_nv12_er_crop_letterbox_to_nchw_rgb_imagenet(
        ctypes.c_void_p(y_ptr),
        ctypes.c_void_p(uv_ptr),
        ctypes.c_void_p(dst_ptr),
        int(full_w),
        int(full_h),
        int(pitch),
        int(crop_left),
        int(crop_top),
        int(crop_w),
        int(crop_h),
        int(dst_w),
        int(dst_h),
        int(new_w),
        int(new_h),
        int(pad_left),
        int(pad_top),
        ctypes.c_void_p(stream_handle),
    )
    if ret != 0:
        raise RuntimeError(
            f"route2_nv12_er_crop_letterbox_to_nchw_rgb_imagenet failed code={ret}"
        )


def nv12_er_crop_to_bgr_hwc(
    y_ptr: int,
    uv_ptr: int,
    dst_ptr: int,
    full_w: int,
    full_h: int,
    pitch: int,
    crop_left: int,
    crop_top: int,
    crop_w: int,
    crop_h: int,
    stream_handle: int = 0,
) -> None:
    lib = _load_lib()
    if lib is None or not hasattr(lib, "route2_nv12_er_crop_to_bgr_hwc"):
        raise RuntimeError(
            "GPU preview 未构建，请重新运行: bash scripts/route2/build_cuda_nv12_rgb.sh"
        )
    ret = lib.route2_nv12_er_crop_to_bgr_hwc(
        ctypes.c_void_p(y_ptr),
        ctypes.c_void_p(uv_ptr),
        ctypes.c_void_p(dst_ptr),
        int(full_w),
        int(full_h),
        int(pitch),
        int(crop_left),
        int(crop_top),
        int(crop_w),
        int(crop_h),
        ctypes.c_void_p(stream_handle),
    )
    if ret != 0:
        raise RuntimeError(f"route2_nv12_er_crop_to_bgr_hwc failed code={ret}")


def gpu_bgr_rotate90_available() -> bool:
    lib = _load_lib()
    return lib is not None and hasattr(lib, "route2_bgr_hwc_rotate90")


def bgr_hwc_rotate90(
    src_ptr: int,
    dst_ptr: int,
    src_h: int,
    src_w: int,
    *,
    clockwise: bool = True,
    stream_handle: int = 0,
) -> None:
    """设备端 BGR HWC 旋转 90°；dst 尺寸为 (src_w, src_h, 3)。"""
    lib = _load_lib()
    if lib is None or not hasattr(lib, "route2_bgr_hwc_rotate90"):
        raise RuntimeError(
            "GPU rotate90 未构建，请重新运行: bash scripts/route2/build_cuda_nv12_rgb.sh"
        )
    ret = lib.route2_bgr_hwc_rotate90(
        ctypes.c_void_p(src_ptr),
        ctypes.c_void_p(dst_ptr),
        int(src_h),
        int(src_w),
        0 if clockwise else 1,
        ctypes.c_void_p(stream_handle),
    )
    if ret != 0:
        raise RuntimeError(f"route2_bgr_hwc_rotate90 failed code={ret}")


def nv12_er_remap_crop_letterbox_to_nchw_rgb(
    y_ptr: int,
    uv_ptr: int,
    map_x_ptr: int,
    map_y_ptr: int,
    dst_ptr: int,
    full_w: int,
    full_h: int,
    pitch: int,
    eye_crop_left: int,
    rect_w: int,
    rect_h: int,
    dst_w: int,
    dst_h: int,
    unpad_w: int,
    unpad_h: int,
    pad_left: int,
    pad_top: int,
    stream_handle: int = 0,
) -> None:
    lib = _load_lib()
    if lib is None or not hasattr(lib, "route2_nv12_er_remap_crop_letterbox_to_nchw_rgb"):
        raise RuntimeError(
            "GPU rectify letterbox 未构建，请重新运行: "
            "bash scripts/route2/build_cuda_nv12_rgb.sh"
        )
    ret = lib.route2_nv12_er_remap_crop_letterbox_to_nchw_rgb(
        ctypes.c_void_p(y_ptr),
        ctypes.c_void_p(uv_ptr),
        ctypes.c_void_p(map_x_ptr),
        ctypes.c_void_p(map_y_ptr),
        ctypes.c_void_p(dst_ptr),
        int(full_w),
        int(full_h),
        int(pitch),
        int(eye_crop_left),
        int(rect_w),
        int(rect_h),
        int(dst_w),
        int(dst_h),
        int(unpad_w),
        int(unpad_h),
        int(pad_left),
        int(pad_top),
        ctypes.c_void_p(stream_handle),
    )
    if ret != 0:
        raise RuntimeError(
            f"route2_nv12_er_remap_crop_letterbox_to_nchw_rgb failed code={ret}"
        )


def nv12_er_remap_bbox_letterbox_to_nchw_rgb_imagenet(
    y_ptr: int,
    uv_ptr: int,
    map_x_ptr: int,
    map_y_ptr: int,
    dst_ptr: int,
    full_w: int,
    full_h: int,
    pitch: int,
    eye_crop_left: int,
    rect_w: int,
    rect_h: int,
    bbox_left: int,
    bbox_top: int,
    bbox_w: int,
    bbox_h: int,
    dst_w: int,
    dst_h: int,
    new_w: int,
    new_h: int,
    pad_left: int,
    pad_top: int,
    stream_handle: int = 0,
) -> None:
    lib = _load_lib()
    if lib is None or not hasattr(
        lib, "route2_nv12_er_remap_bbox_letterbox_to_nchw_rgb_imagenet"
    ):
        raise RuntimeError(
            "GPU rectify ViTPose letterbox 未构建，请重新运行: "
            "bash scripts/route2/build_cuda_nv12_rgb.sh"
        )
    ret = lib.route2_nv12_er_remap_bbox_letterbox_to_nchw_rgb_imagenet(
        ctypes.c_void_p(y_ptr),
        ctypes.c_void_p(uv_ptr),
        ctypes.c_void_p(map_x_ptr),
        ctypes.c_void_p(map_y_ptr),
        ctypes.c_void_p(dst_ptr),
        int(full_w),
        int(full_h),
        int(pitch),
        int(eye_crop_left),
        int(rect_w),
        int(rect_h),
        int(bbox_left),
        int(bbox_top),
        int(bbox_w),
        int(bbox_h),
        int(dst_w),
        int(dst_h),
        int(new_w),
        int(new_h),
        int(pad_left),
        int(pad_top),
        ctypes.c_void_p(stream_handle),
    )
    if ret != 0:
        raise RuntimeError(
            "route2_nv12_er_remap_bbox_letterbox_to_nchw_rgb_imagenet "
            f"failed code={ret}"
        )
