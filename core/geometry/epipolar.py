"""双目 2D 极线约束：对称极线距离 + 校正后垂直偏移。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.geometry.calib_model import StereoCalibration


def fundamental_matrix_from_calib(calib: StereoCalibration) -> np.ndarray:
    """左→右基本矩阵 F（像素坐标，OpenCV 约定）。"""
    t = np.asarray(calib.T, dtype=np.float64).reshape(3)
    tx = np.array(
        [
            [0.0, -t[2], t[1]],
            [t[2], 0.0, -t[0]],
            [-t[1], t[0], 0.0],
        ],
        dtype=np.float64,
    )
    e = tx @ np.asarray(calib.R, dtype=np.float64)
    k1_inv = np.linalg.inv(np.asarray(calib.K1, dtype=np.float64))
    k2_inv_t = np.linalg.inv(np.asarray(calib.K2, dtype=np.float64)).T
    return k2_inv_t @ e @ k1_inv


def _homog_xy(xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    return np.hstack([pts, ones])


def symmetric_epipolar_distance_px(
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    f: np.ndarray,
) -> np.ndarray:
    """逐点对称极线距离（像素）。pts: (N,2)。"""
    x1 = _homog_xy(pts_left)
    x2 = _homog_xy(pts_right)
    f = np.asarray(f, dtype=np.float64)
    l2 = (f @ x1.T).T
    l1 = (f.T @ x2.T).T
    num2 = np.abs(np.sum(x2 * l2, axis=1))
    den2 = np.sqrt(l2[:, 0] ** 2 + l2[:, 1] ** 2)
    num1 = np.abs(np.sum(x1 * l1, axis=1))
    den1 = np.sqrt(l1[:, 0] ** 2 + l1[:, 1] ** 2)
    d2 = num2 / np.maximum(den2, 1e-9)
    d1 = num1 / np.maximum(den1, 1e-9)
    return 0.5 * (d1 + d2)


def undistort_to_rectified(
    pts: np.ndarray,
    *,
    k: np.ndarray,
    d: np.ndarray,
    r: np.ndarray,
    p: np.ndarray,
) -> np.ndarray:
    """像素 → 校正平面 (u,v)。"""
    xy = np.asarray(pts, dtype=np.float64).reshape(-1, 1, 2)
    r = np.asarray(r, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    k = np.asarray(k, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64).ravel()
    out = cv2.undistortPoints(xy, k, d, None, r, p)
    return out.reshape(-1, 2)


@dataclass(frozen=True)
class EpipolarJointMetrics:
    dy_rect_px: float
    dx_rect_px: float
    epi_sym_px: float
    conf: float


@dataclass(frozen=True)
class EpipolarFrameMetrics:
    capture_seq: int
    joint_count: int
    mean_dy_rect_px: float
    mean_abs_dy_rect_px: float
    p95_abs_dy_rect_px: float
    mean_epi_sym_px: float
    p95_epi_sym_px: float
    per_joint: dict[int, EpipolarJointMetrics]


def measure_stereo_2d_epipolar(
    kpts_left: np.ndarray,
    kpts_right: np.ndarray,
    calib: StereoCalibration,
    *,
    min_conf: float = 0.25,
    coords_rectified: bool = False,
) -> EpipolarFrameMetrics | None:
    """匹配左右 2D 点，返回极线/垂直偏移指标。

    coords_rectified=True：输入已是校正图像素坐标（方案 B infer 产出）。
    """
    kl = np.asarray(kpts_left, dtype=np.float64)
    kr = np.asarray(kpts_right, dtype=np.float64)
    if kl.ndim != 2 or kr.ndim != 2 or kl.shape[0] != kr.shape[0]:
        return None

    n = kl.shape[0]
    conf_l = kl[:, 2] if kl.shape[1] >= 3 else np.ones(n, dtype=np.float64)
    conf_r = kr[:, 2] if kr.shape[1] >= 3 else np.ones(n, dtype=np.float64)
    valid = (
        (conf_l >= float(min_conf))
        & (conf_r >= float(min_conf))
        & (kl[:, 0] > 1.0)
        & (kr[:, 0] > 1.0)
    )
    if not np.any(valid):
        return None

    idx = np.flatnonzero(valid)
    pl = kl[idx, :2]
    pr = kr[idx, :2]

    if coords_rectified:
        pl_ud = pl
        pr_ud = pr
        pl_rect = pl
        pr_rect = pr
    else:
        pl_ud = cv2.undistortPoints(
            pl.reshape(-1, 1, 2),
            calib.K1,
            calib.D1,
            None,
            calib.K1,
        ).reshape(-1, 2)
        pr_ud = cv2.undistortPoints(
            pr.reshape(-1, 1, 2),
            calib.K2,
            calib.D2,
            None,
            calib.K2,
        ).reshape(-1, 2)
        pl_rect = undistort_to_rectified(
            pl, k=calib.K1, d=calib.D1, r=calib.R1, p=calib.P1
        )
        pr_rect = undistort_to_rectified(
            pr, k=calib.K2, d=calib.D2, r=calib.R2, p=calib.P2
        )

    f = fundamental_matrix_from_calib(calib)
    epi = symmetric_epipolar_distance_px(pl_ud, pr_ud, f)
    dy = pr_rect[:, 1] - pl_rect[:, 1]
    dx = pr_rect[:, 0] - pl_rect[:, 0]

    per_joint: dict[int, EpipolarJointMetrics] = {}
    for k, ji in enumerate(idx):
        c = float(min(conf_l[ji], conf_r[ji]))
        per_joint[int(ji)] = EpipolarJointMetrics(
            dy_rect_px=float(dy[k]),
            dx_rect_px=float(dx[k]),
            epi_sym_px=float(epi[k]),
            conf=c,
        )

    abs_dy = np.abs(dy)
    return EpipolarFrameMetrics(
        capture_seq=-1,
        joint_count=int(len(idx)),
        mean_dy_rect_px=float(np.mean(dy)),
        mean_abs_dy_rect_px=float(np.mean(abs_dy)),
        p95_abs_dy_rect_px=float(np.percentile(abs_dy, 95)),
        mean_epi_sym_px=float(np.mean(epi)),
        p95_epi_sym_px=float(np.percentile(epi, 95)),
        per_joint=per_joint,
    )
