"""校正双目纹理深度：SGBM 视差 → 米制 depth（左相机 OpenCV 系）。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.geometry.calib_model import StereoCalibration
from core.geometry.stereo_rectify import StereoRectifier, rectified_to_left_cam_opencv


@dataclass(frozen=True)
class StereoSgbmConfig:
    min_disparity: int = 0
    num_disparities: int = 256
    block_size: int = 5
    p1: int = 0
    p2: int = 0
    disp12_max_diff: int = 1
    pre_filter_cap: int = 63
    uniqueness_ratio: int = 10
    speckle_window_size: int = 100
    speckle_range: int = 2
    mode: int = cv2.STEREO_SGBM_MODE_SGBM_3WAY


@dataclass(frozen=True)
class StereoTextureDepthResult:
    disparity: np.ndarray | None
    depth_m: np.ndarray | None
    points3d_rect: np.ndarray | None
    valid_mask: np.ndarray | None
    sample_z_left: np.ndarray | None = None
    sample_uv_rect: np.ndarray | None = None
    sample_kpts3d_left: np.ndarray | None = None


def make_stereo_matcher(cfg: StereoSgbmConfig) -> cv2.StereoSGBM:
    bs = int(cfg.block_size)
    if bs % 2 == 0:
        bs += 1
    nd = int(cfg.num_disparities)
    if nd % 16 != 0:
        nd = (nd // 16 + 1) * 16
    return cv2.StereoSGBM_create(
        minDisparity=int(cfg.min_disparity),
        numDisparities=nd,
        blockSize=bs,
        P1=int(cfg.p1 or 8 * 3 * bs * bs),
        P2=int(cfg.p2 or 6 * 3 * bs * bs),
        disp12MaxDiff=int(cfg.disp12_max_diff),
        preFilterCap=int(cfg.pre_filter_cap),
        uniquenessRatio=int(cfg.uniqueness_ratio),
        speckleWindowSize=int(cfg.speckle_window_size),
        speckleRange=int(cfg.speckle_range),
        mode=int(cfg.mode),
    )


def disparity_to_depth_rectified_m(
    disp: np.ndarray,
    calib: StereoCalibration,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """视差（像素）→ 校正左相机系深度（米）。"""
    sc = float(scale)
    fx = float(calib.P1[0, 0]) * sc
    baseline = float(calib.baseline_m)
    depth = np.full(disp.shape, np.nan, dtype=np.float32)
    mask = np.asarray(disp, dtype=np.float32) > 0.5
    depth[mask] = (fx * baseline / disp[mask]).astype(np.float32)
    return depth


def rectified_uvz_to_left_cam(
    uv: np.ndarray,
    z_rect: np.ndarray,
    calib: StereoCalibration,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """N 点：校正图像素 + 校正系 Z → 原左相机系 (X,Y,Z)。"""
    sc = float(scale)
    fx = float(calib.P1[0, 0]) * sc
    fy = float(calib.P1[1, 1]) * sc
    cx = float(calib.P1[0, 2]) * sc
    cy = float(calib.P1[1, 2]) * sc
    r1 = np.asarray(calib.R1, dtype=np.float64)

    u = np.asarray(uv, dtype=np.float64)[:, 0]
    v = np.asarray(uv, dtype=np.float64)[:, 1]
    z = np.asarray(z_rect, dtype=np.float64)
    x_r = (u - cx) * z / fx
    y_r = (v - cy) * z / fy
    pts_rect = np.stack([x_r, y_r, z], axis=1)
    return rectified_to_left_cam_opencv(pts_rect, r1)


def compute_stereo_texture_depth(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    calib: StereoCalibration,
    *,
    rectifier: StereoRectifier | None = None,
    matcher: cv2.StereoSGBM | None = None,
    cfg: StereoSgbmConfig | None = None,
    scale: float = 1.0,
) -> StereoTextureDepthResult:
    """BGR 左右眼 → 左相机系 depth（米）；可选 scale 降采样加速。"""
    rect = rectifier or StereoRectifier.from_calib(calib)
    left_r, right_r = rect.rectify_pair(left_bgr, right_bgr)

    sc = float(scale)
    if sc > 0.0 and sc != 1.0:
        left_r = cv2.resize(left_r, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        right_r = cv2.resize(right_r, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)

    gray_l = cv2.cvtColor(left_r, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(right_r, cv2.COLOR_BGR2GRAY)

    sgbm_cfg = cfg or StereoSgbmConfig()
    sgbm = matcher or make_stereo_matcher(sgbm_cfg)
    disp_raw = sgbm.compute(gray_l, gray_r)
    disp = disp_raw.astype(np.float32) / 16.0
    depth_rect = disparity_to_depth_rectified_m(disp, calib, scale=sc)
    valid = (
        np.isfinite(depth_rect)
        & (depth_rect > 0.05)
        & (depth_rect < 50.0)
    )

    depth_left = np.full_like(depth_rect, np.nan, dtype=np.float32)
    if np.any(valid):
        ys, xs = np.nonzero(valid)
        uv = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
        pts_left = rectified_uvz_to_left_cam(
            uv, depth_rect[valid].astype(np.float64), calib, scale=sc
        )
        depth_left[valid] = pts_left[:, 2].astype(np.float32)

    pts3d_rect = np.dstack([depth_rect, depth_rect, depth_rect])

    return StereoTextureDepthResult(
        disparity=disp,
        depth_m=depth_left,
        points3d_rect=pts3d_rect,
        valid_mask=valid,
    )


def distorted_pixels_to_rectified(
    uv: np.ndarray,
    calib: StereoCalibration,
) -> np.ndarray:
    """N×2 畸变左图像素 → N×2 校正图像素。"""
    pts = np.asarray(uv, dtype=np.float64).reshape(-1, 1, 2)
    k1 = np.asarray(calib.K1, dtype=np.float64)
    d1 = np.asarray(calib.D1, dtype=np.float64).ravel()
    r1 = np.asarray(calib.R1, dtype=np.float64)
    p1 = np.asarray(calib.P1, dtype=np.float64)
    out = cv2.undistortPoints(pts, k1, d1, R=r1, P=p1)
    return out.reshape(-1, 2)


def sample_disparity_bilinear_q10_5(
    disp_q10: np.ndarray,
    uv: np.ndarray,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """S16 Q10.5 视差图 + 校正图像素 → 双线性采样（无效为 nan）。"""
    h, w = disp_q10.shape[:2]
    sc = float(scale)
    u = np.asarray(uv, dtype=np.float64)[:, 0] * sc
    v = np.asarray(uv, dtype=np.float64)[:, 1] * sc
    out = np.full(u.shape[0], np.nan, dtype=np.float32)
    for i in range(u.shape[0]):
        ui, vi = float(u[i]), float(v[i])
        if not np.isfinite(ui) or not np.isfinite(vi):
            continue
        if ui < 0 or vi < 0 or ui >= w - 1 or vi >= h - 1:
            continue
        x0, y0 = int(ui), int(vi)
        ax, ay = ui - x0, vi - y0
        patch = disp_q10[y0 : y0 + 2, x0 : x0 + 2].astype(np.float32)
        val = (
            (1 - ax) * (1 - ay) * patch[0, 0]
            + ax * (1 - ay) * patch[0, 1]
            + (1 - ax) * ay * patch[1, 0]
            + ax * ay * patch[1, 1]
        )
        if val > 16.0:
            out[i] = val
    return out


def disparity_q_to_depth_z(
    disp_q10_5: np.ndarray,
    calib: StereoCalibration,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """Q10.5 视差采样值 → 左相机系 Z（米）；无效为 nan。"""
    sc = float(scale)
    fx = float(calib.P1[0, 0]) * sc
    baseline = float(calib.baseline_m)
    disp = np.asarray(disp_q10_5, dtype=np.float64) / 32.0
    z = np.full(disp.shape, np.nan, dtype=np.float64)
    ok = np.isfinite(disp) & (disp > 0.5)
    z[ok] = fx * baseline / disp[ok]
    return z


def sample_depth_bilinear(
    depth: np.ndarray,
    uv: np.ndarray,
    *,
    scale: float = 1.0,
) -> np.ndarray:
    """校正/与 depth 同分辨率的像素坐标 → 深度采样（无效为 nan）。"""
    h, w = depth.shape[:2]
    sc = float(scale)
    u = np.asarray(uv, dtype=np.float64)[:, 0] * sc
    v = np.asarray(uv, dtype=np.float64)[:, 1] * sc
    out = np.full(u.shape[0], np.nan, dtype=np.float64)
    for i in range(u.shape[0]):
        ui, vi = float(u[i]), float(v[i])
        if not np.isfinite(ui) or not np.isfinite(vi):
            continue
        if ui < 0 or vi < 0 or ui >= w - 1 or vi >= h - 1:
            continue
        x0, y0 = int(ui), int(vi)
        ax, ay = ui - x0, vi - y0
        z = (
            (1 - ax) * (1 - ay) * depth[y0, x0]
            + ax * (1 - ay) * depth[y0, x0 + 1]
            + (1 - ax) * ay * depth[y0 + 1, x0]
            + ax * ay * depth[y0 + 1, x0 + 1]
        )
        if np.isfinite(z) and z > 0.05:
            out[i] = float(z)
    return out


def depth_to_colormap_bgr(depth_m: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """可视化：近红远蓝。"""
    vis = np.zeros((*depth_m.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return vis
    z = depth_m.copy()
    z[~valid] = np.nan
    zmin = float(np.nanpercentile(z, 5))
    zmax = float(np.nanpercentile(z, 95))
    if zmax <= zmin + 1e-3:
        zmax = zmin + 1.0
    norm = np.clip((z - zmin) / (zmax - zmin), 0.0, 1.0)
    norm = np.nan_to_num(norm, nan=0.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    norm_u8[~valid] = 0
    cm = cv2.applyColorMap(norm_u8, cv2.COLORMAP_TURBO)
    cm[~valid] = 0
    return cm
