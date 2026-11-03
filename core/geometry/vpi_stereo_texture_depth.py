"""Jetson VPI 纹理立体视差 → 米制 depth（左相机 OpenCV 系）。

支持 backend：
- CUDA：pitch-linear 灰度，census+SGM
- OFA|PVA|VIC：需 block-linear Y8（经 VIC 从 Y8 转换），eSGM + PVA 后处理

`full_depth=False` 时仅在 `sample_uv_rect` 处读视差，跳过全图 disp.cpu() 与 depth 换算。
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.geometry.calib_model import StereoCalibration
from core.geometry.stereo_rectify import StereoRectifier
from core.geometry.stereo_texture_depth import (
    StereoTextureDepthResult,
    disparity_q_to_depth_z,
    disparity_to_depth_rectified_m,
    rectified_uvz_to_left_cam,
    sample_disparity_bilinear_q10_5,
)

try:
    import vpi
except ImportError:  # pragma: no cover
    vpi = None  # type: ignore


VPI_BACKEND_CUDA = "cuda"
VPI_BACKEND_OFA_PVA_VIC = "ofa_pva_vic"


def resolve_vpi_backend(name: str):
    """解析 benchmark/CLI 用的 backend 名 → vpi.Backend。"""
    _require_vpi()
    key = str(name).strip().lower().replace("-", "_")
    if key in ("cuda", "gpu"):
        return vpi.Backend.CUDA
    if key in ("pva", "ofa", "ofa_pva_vic", "ofa_pva", "pva_vic"):
        return vpi.Backend.OFA | vpi.Backend.PVA | vpi.Backend.VIC
    raise ValueError(f"未知 VPI backend: {name!r}（支持 cuda / ofa_pva_vic）")


_OFA_PVA_VIC = None


def _ofa_pva_vic_backend():
    global _OFA_PVA_VIC
    _require_vpi()
    if _OFA_PVA_VIC is None:
        _OFA_PVA_VIC = vpi.Backend.OFA | vpi.Backend.PVA | vpi.Backend.VIC
    return _OFA_PVA_VIC


def vpi_backend_label(backend) -> str:
    _require_vpi()
    if backend == vpi.Backend.CUDA:
        return "vpi.CUDA"
    if backend == _ofa_pva_vic_backend():
        return "vpi.OFA|PVA|VIC"
    return str(backend)


@dataclass(frozen=True)
class VpiStereoConfig:
    backend: object | None = None  # vpi.Backend.CUDA
    maxdisp: int = 256
    window: int = 5
    includediagonals: bool = False
    downscale: int = 1
    numpasses: int = 3
    confthreshold: int = 32767
    quality: int = 6


def _require_vpi() -> None:
    if vpi is None:
        raise ImportError("需要 python3-vpi3（Jetson VPI）")


def _needs_block_linear_y8(backend) -> bool:
    _require_vpi()
    return backend == _ofa_pva_vic_backend()


def _to_stereo_input(gray: np.ndarray, backend):
    """OFA|PVA|VIC 需要 Y8_BL；CUDA 用 pitch-linear Y8。"""
    img = vpi.asimage(gray, vpi.Format.Y8)
    if _needs_block_linear_y8(backend):
        with vpi.Backend.VIC:
            img = img.convert(vpi.Format.Y8_BL, backend=vpi.Backend.VIC)
    return img


def _build_full_depth_result(
    disp_q: np.ndarray,
    calib: StereoCalibration,
    *,
    scale: float,
) -> StereoTextureDepthResult:
    sc = float(scale)
    depth_rect = disparity_to_depth_rectified_m(disp_q, calib, scale=sc)
    valid = (
        np.isfinite(depth_rect)
        & (depth_rect > 0.05)
        & (depth_rect < 50.0)
        & (disp_q > 0.5)
    )

    depth_left = np.full_like(depth_rect, np.nan, dtype=np.float32)
    if np.any(valid):
        ys, xs = np.nonzero(valid)
        uv = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        pts_left = rectified_uvz_to_left_cam(
            uv, depth_rect[valid].astype(np.float64), calib, scale=sc
        )
        depth_left[valid] = pts_left[:, 2].astype(np.float32)

    return StereoTextureDepthResult(
        disparity=disp_q,
        depth_m=depth_left,
        points3d_rect=np.dstack([depth_rect, depth_rect, depth_rect]),
        valid_mask=valid,
    )


def _build_sparse_depth_result(
    disp_img,
    sample_uv_rect: np.ndarray,
    calib: StereoCalibration,
    *,
    scale: float,
) -> StereoTextureDepthResult:
    uv = np.asarray(sample_uv_rect, dtype=np.float64).reshape(-1, 2)
    with disp_img.rlock_cpu() as disp_raw:
        disp_q_samples = sample_disparity_bilinear_q10_5(
            disp_raw, uv, scale=scale
        )

    disp_q = disp_q_samples.astype(np.float32) / 32.0
    z_rect = disparity_q_to_depth_z(disp_q_samples, calib, scale=scale)
    z_left = np.full(uv.shape[0], np.nan, dtype=np.float32)
    kpts3d = np.full((uv.shape[0], 3), np.nan, dtype=np.float32)
    ok = np.isfinite(z_rect) & (z_rect > 0.05) & (z_rect < 50.0)
    if np.any(ok):
        pts_left = rectified_uvz_to_left_cam(
            uv[ok], z_rect[ok], calib, scale=scale
        )
        kpts3d[ok] = pts_left.astype(np.float32)
        z_left[ok] = pts_left[:, 2].astype(np.float32)

    return StereoTextureDepthResult(
        disparity=None,
        depth_m=None,
        points3d_rect=None,
        valid_mask=None,
        sample_z_left=z_left,
        sample_uv_rect=uv,
        sample_kpts3d_left=kpts3d,
    )


def compute_vpi_stereo_texture_depth(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    calib: StereoCalibration,
    *,
    rectifier: StereoRectifier | None = None,
    cfg: VpiStereoConfig | None = None,
    scale: float = 1.0,
    sample_uv_rect: np.ndarray | None = None,
    full_depth: bool = True,
) -> StereoTextureDepthResult:
    """校正灰度图 → VPI stereodisp → 左相机 depth（米）。

    ``full_depth=False`` 且提供 ``sample_uv_rect`` 时，仅在关键点采样视差，
    避免全图 ``disp.cpu()`` 与稠密 depth 换算。
    """
    _require_vpi()
    cfg = cfg or VpiStereoConfig(backend=vpi.Backend.CUDA)
    backend = cfg.backend or vpi.Backend.CUDA
    sparse = not full_depth and sample_uv_rect is not None

    rect = rectifier or StereoRectifier.from_calib(calib)
    left_r, right_r = rect.rectify_pair(left_bgr, right_bgr)

    sc = float(scale)
    if sc > 0.0 and sc != 1.0:
        left_r = cv2.resize(left_r, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        right_r = cv2.resize(right_r, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)

    gray_l = cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY)

    stereo_kwargs: dict = {
        "backend": backend,
        "maxdisp": int(cfg.maxdisp),
        "window": int(cfg.window),
        "downscale": int(cfg.downscale),
        "confthreshold": int(cfg.confthreshold),
    }
    if _needs_block_linear_y8(backend):
        stereo_kwargs["numpasses"] = int(cfg.numpasses)
    else:
        stereo_kwargs["includediagonals"] = bool(cfg.includediagonals)
        stereo_kwargs["quality"] = int(cfg.quality)

    with backend:
        img_l = _to_stereo_input(gray_l, backend)
        img_r = _to_stereo_input(gray_r, backend)
        disp_img = vpi.stereodisp(img_l, img_r, **stereo_kwargs)
        if sparse:
            return _build_sparse_depth_result(
                disp_img,
                sample_uv_rect,
                calib,
                scale=sc,
            )
        disp_q = disp_img.cpu().astype(np.float32) / 32.0

    return _build_full_depth_result(disp_q, calib, scale=sc)
