"""把横幅单目标定变换到与 eye_mount 竖图关键点同一像素系。

与 ``apply_eye_mount_rotate`` 像素约定一致：
- cw90:  (x, y) → (H-1-y, x)，新尺寸 W'=H, H'=W
- ccw90: (x, y) → (y, W-1-x)，新尺寸 W'=H, H'=W

畸变 D（OpenCV Brown/有理 + 薄棱镜 + 倾斜）按归一化平面 90° 严格变换：
径向 k 不变；切向 / 薄棱镜 / τ 随轴系旋转。也可用 ``zero_distortion=True`` 置零。
K/R/T 按 upright 关键点系适配，并按新尺寸重跑 stereoRectify。
"""

from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np

from core.geometry.calib_model import StereoCalibration
from core.geometry.eye_mount_rotate import EyeMountRotate, parse_eye_mount_rotate


def _r90_cw() -> np.ndarray:
    """相机系：X'=-Y, Y'=X, Z'=Z（配合图像 cw90）。"""
    return np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _r90_ccw() -> np.ndarray:
    """相机系：X'=Y, Y'=-X, Z'=Z（配合图像 ccw90）。"""
    return np.array(
        [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _transform_K(K: np.ndarray, *, mode: EyeMountRotate, width: int, height: int) -> np.ndarray:
    """横幅 K → 竖图 K（主点/焦距随像素 90° 变换）。"""
    k = np.asarray(K, dtype=np.float64).reshape(3, 3)
    fx, fy = float(k[0, 0]), float(k[1, 1])
    cx, cy = float(k[0, 2]), float(k[1, 2])
    w, h = int(width), int(height)
    out = np.eye(3, dtype=np.float64)
    if mode == "cw90":
        # x'=H-1-y, y'=x  →  fx'=fy, fy'=fx, cx'=H-1-cy, cy'=cx
        out[0, 0] = fy
        out[1, 1] = fx
        out[0, 2] = float(h - 1) - cy
        out[1, 2] = cx
    elif mode == "ccw90":
        # x'=y, y'=W-1-x  →  fx'=fy, fy'=fx, cx'=cy, cy'=W-1-cx
        out[0, 0] = fy
        out[1, 1] = fx
        out[0, 2] = cy
        out[1, 2] = float(w - 1) - cx
    else:
        return k.copy()
    return out


def transform_distortion_for_eye_rotate(
    D: np.ndarray,
    mode: EyeMountRotate | str,
) -> np.ndarray:
    """把 OpenCV 畸变向量变到与 eye_mount 90° 同一归一化平面。

    布局（长度 4/5/8/12/14 均支持，缺项视为 0）：
      [k1,k2,p1,p2,k3, k4,k5,k6, s1,s2,s3,s4, τx,τy]

    - 径向 k1…k6：r 不变量，保持
    - 切向 (p1,p2)：cw → (p2,-p1)；ccw → (-p2,p1)
    - 薄棱镜 (s1..s4)：随 (Δx,Δy) 同式旋转
    - 倾斜 (τx,τy)：同切向二维向量旋转
    """
    rot = parse_eye_mount_rotate(mode if isinstance(mode, str) else str(mode))
    d_in = np.asarray(D, dtype=np.float64).ravel()
    n = int(d_in.size)
    if n == 0 or rot == "none":
        return d_in.copy()

    d = np.zeros(14, dtype=np.float64)
    d[: min(n, 14)] = d_in[: min(n, 14)]

    p1, p2 = float(d[2]), float(d[3])
    s1, s2, s3, s4 = float(d[8]), float(d[9]), float(d[10]), float(d[11])
    tx, ty = float(d[12]), float(d[13])

    if rot == "cw90":
        # 归一化： (x',y')=(-y,x)，畸变位移 (Δx',Δy')=(-Δy,Δx)
        d[2], d[3] = p2, -p1
        d[8], d[9], d[10], d[11] = -s3, -s4, s1, s2
        d[12], d[13] = ty, -tx
    else:  # ccw90
        # (x',y')=(y,-x)，(Δx',Δy')=(Δy,-Δx)
        d[2], d[3] = -p2, p1
        d[8], d[9], d[10], d[11] = s3, s4, -s1, -s2
        d[12], d[13] = -ty, tx

    return d[:n].copy()


def _refresh_rectification(
    *,
    K1: np.ndarray,
    K2: np.ndarray,
    D1: np.ndarray,
    D2: np.ndarray,
    R: np.ndarray,
    T: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """按新尺寸与当前 D 重算 R1/R2/P1/P2/Q。"""
    r1, r2, p1, p2, q, _, _ = cv2.stereoRectify(
        np.asarray(K1, dtype=np.float64),
        np.asarray(D1, dtype=np.float64).ravel(),
        np.asarray(K2, dtype=np.float64),
        np.asarray(D2, dtype=np.float64).ravel(),
        (int(width), int(height)),
        np.asarray(R, dtype=np.float64),
        np.asarray(T, dtype=np.float64).reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    return (
        np.asarray(r1, dtype=np.float64),
        np.asarray(r2, dtype=np.float64),
        np.asarray(p1, dtype=np.float64),
        np.asarray(p2, dtype=np.float64),
        np.asarray(q, dtype=np.float64),
    )


def adapt_stereo_calib_for_eye_rotate(
    calib: StereoCalibration,
    mode: EyeMountRotate | str,
    *,
    zero_distortion: bool = False,
) -> StereoCalibration:
    """横幅标定 → 与 upright 关键点对齐的标定。mode=none 时原样返回。

    默认对 D 做 90° **严格变换**；``zero_distortion=True`` 时 D 置零。
    """
    rot = parse_eye_mount_rotate(mode if isinstance(mode, str) else str(mode))
    if rot == "none":
        return calib

    w0 = int(calib.image_width)
    h0 = int(calib.image_height)
    w1, h1 = h0, w0  # upright: 1080x1920 from 1920x1080

    r90 = _r90_cw() if rot == "cw90" else _r90_ccw()
    k1 = _transform_K(calib.K1, mode=rot, width=w0, height=h0)
    k2 = _transform_K(calib.K2, mode=rot, width=w0, height=h0)
    r = r90 @ np.asarray(calib.R, dtype=np.float64) @ r90.T
    t = r90 @ np.asarray(calib.T, dtype=np.float64).reshape(3, 1)

    if zero_distortion:
        d1 = np.zeros_like(np.asarray(calib.D1, dtype=np.float64).ravel())
        d2 = np.zeros_like(np.asarray(calib.D2, dtype=np.float64).ravel())
        d_tag = "0"
    else:
        d1 = transform_distortion_for_eye_rotate(calib.D1, rot)
        d2 = transform_distortion_for_eye_rotate(calib.D2, rot)
        d_tag = "strict_rot"

    r1, r2, p1, p2, q = _refresh_rectification(
        K1=k1,
        K2=k2,
        D1=d1,
        D2=d2,
        R=r,
        T=t,
        width=w1,
        height=h1,
    )

    note = (calib.notes or "").strip()
    tag = f"eye_rotate={rot} K/R/T/D adapted D={d_tag}"
    notes = f"{note}; {tag}" if note else tag

    return replace(
        calib,
        image_width=w1,
        image_height=h1,
        K1=k1,
        K2=k2,
        D1=d1,
        D2=d2,
        R=r,
        T=t,
        R1=r1,
        R2=r2,
        P1=p1,
        P2=p2,
        Q=q,
        notes=notes,
    )
