"""左/右相机系 3D 点 → 像素（离线 preview / 重投影验收）。"""

from __future__ import annotations

import cv2
import numpy as np

from core.geometry.calib_model import StereoCalibration

_RVEC_LEFT = np.zeros((3, 1), dtype=np.float64)
_TVEC_LEFT = np.zeros((3, 1), dtype=np.float64)


def _right_extrinsics(calib: StereoCalibration) -> tuple[np.ndarray, np.ndarray]:
    rvec_r, _ = cv2.Rodrigues(np.asarray(calib.R, dtype=np.float64))
    tvec_r = np.asarray(calib.T, dtype=np.float64).reshape(3, 1)
    return rvec_r, tvec_r


def project_points3d_to_left2d(
    pts3d: np.ndarray,
    calib: StereoCalibration,
) -> np.ndarray:
    """(N,3) 左相机米制 → (N,2) 像素。"""
    pts = np.asarray(pts3d, dtype=np.float64).reshape(-1, 1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    uv, _ = cv2.projectPoints(pts, _RVEC_LEFT, _TVEC_LEFT, calib.K1, calib.D1)
    return uv.reshape(-1, 2).astype(np.float32)


def project_points3d_to_right2d(
    pts3d: np.ndarray,
    calib: StereoCalibration,
) -> np.ndarray:
    """(N,3) 左相机米制 → 右目 (N,2) 像素。"""
    pts = np.asarray(pts3d, dtype=np.float64).reshape(-1, 1, 3)
    if pts.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    rvec_r, tvec_r = _right_extrinsics(calib)
    uv, _ = cv2.projectPoints(pts, rvec_r, tvec_r, calib.K2, calib.D2)
    return uv.reshape(-1, 2).astype(np.float32)


def kpts4_to_det2d(kpts4: np.ndarray, uv: np.ndarray) -> dict:
    """(N,4) + (N,2) → det keypoints (N,3)。"""
    k = np.asarray(kpts4, dtype=np.float32).reshape(-1, 4)
    u = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    n = min(k.shape[0], u.shape[0])
    out = np.zeros((n, 3), dtype=np.float32)
    out[:, :2] = u[:n]
    out[:, 2] = k[:n, 3]
    return {"keypoints": out}


def project_kpts4_to_left2d(
    kpts4: np.ndarray | None,
    calib: StereoCalibration,
) -> dict | None:
    """kpts4 (N,4) 左相机米制 → det_left 形 keypoints (N,3)。"""
    if kpts4 is None:
        return None
    k = np.asarray(kpts4, dtype=np.float64)
    if k.ndim != 2 or k.shape[1] < 4 or k.shape[0] == 0:
        return None
    uv = project_points3d_to_left2d(k[:, :3], calib)
    return kpts4_to_det2d(k, uv)


def project_kpts4_stereo(
    kpts4: np.ndarray | None,
    calib: StereoCalibration,
) -> tuple[dict | None, dict | None]:
    """kpts4 (N,4) → (det_left, det_right)。"""
    if kpts4 is None:
        return None, None
    k = np.asarray(kpts4, dtype=np.float64)
    if k.ndim != 2 or k.shape[1] < 4 or k.shape[0] == 0:
        return None, None
    xyz = k[:, :3]
    return (
        kpts4_to_det2d(k, project_points3d_to_left2d(xyz, calib)),
        kpts4_to_det2d(k, project_points3d_to_right2d(xyz, calib)),
    )


def project_kpts3d_to_left2d(
    kpts3d: np.ndarray | None,
    calib: StereoCalibration,
) -> dict | None:
    """kpts3d (17,4) 左相机米制 → det_left 形 keypoints (17,3)。"""
    return project_kpts4_to_left2d(kpts3d, calib)
