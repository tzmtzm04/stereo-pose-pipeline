"""立体校正：initUndistortRectifyMap + remap（方案 B：在校正图上 infer）。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.geometry.calib_model import StereoCalibration


@dataclass(frozen=True)
class StereoRectifier:
    """预计算 remap，将各眼原图 → stereoRectify 校正平面。"""

    width: int
    height: int
    map1x: np.ndarray
    map1y: np.ndarray
    map2x: np.ndarray
    map2y: np.ndarray
    r1: np.ndarray

    @classmethod
    def from_calib(cls, calib: StereoCalibration) -> StereoRectifier:
        w = int(calib.image_width)
        h = int(calib.image_height)
        size = (w, h)
        k1 = np.asarray(calib.K1, dtype=np.float64)
        d1 = np.asarray(calib.D1, dtype=np.float64).ravel()
        k2 = np.asarray(calib.K2, dtype=np.float64)
        d2 = np.asarray(calib.D2, dtype=np.float64).ravel()
        r1 = np.asarray(calib.R1, dtype=np.float64)
        r2 = np.asarray(calib.R2, dtype=np.float64)
        p1 = np.asarray(calib.P1, dtype=np.float64)
        p2 = np.asarray(calib.P2, dtype=np.float64)
        map1x, map1y = cv2.initUndistortRectifyMap(
            k1, d1, r1, p1, size, cv2.CV_32FC1
        )
        map2x, map2y = cv2.initUndistortRectifyMap(
            k2, d2, r2, p2, size, cv2.CV_32FC1
        )
        return cls(
            width=w,
            height=h,
            map1x=map1x,
            map1y=map1y,
            map2x=map2x,
            map2y=map2y,
            r1=r1,
        )

    def rectify_pair(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        left_r = cv2.remap(
            left_bgr,
            self.map1x,
            self.map1y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        right_r = cv2.remap(
            right_bgr,
            self.map2x,
            self.map2y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        return left_r, right_r


def rectified_to_left_cam_opencv(xyz: np.ndarray, r1: np.ndarray) -> np.ndarray:
    """校正左相机系 3D → 原左相机 OpenCV 系（与现有 jsonl / 显示一致）。"""
    pts = np.asarray(xyz, dtype=np.float64)
    rot = np.asarray(r1, dtype=np.float64).T
    if pts.ndim == 1:
        return (rot @ pts.reshape(3, 1)).reshape(3).astype(np.float32)
    out = (rot @ pts.reshape(-1, 3).T).T
    return out.astype(np.float32)


def sample_rectified_to_distorted_pixel(
    map_x: np.ndarray,
    map_y: np.ndarray,
    u: float,
    v: float,
) -> tuple[float, float]:
    """校正图像素 (u,v) → 原畸变图像素（initUndistortRectifyMap 逆查表）。"""
    h, w = map_x.shape
    if u <= 0 or v <= 0 or not np.isfinite(u) or not np.isfinite(v):
        return 0.0, 0.0
    ui = float(np.clip(u, 0.0, w - 1.001))
    vi = float(np.clip(v, 0.0, h - 1.001))
    x0, y0 = int(ui), int(vi)
    x1 = min(x0 + 1, w - 1)
    y1 = min(y0 + 1, h - 1)
    ax, ay = ui - x0, vi - y0

    def _bilinear(m: np.ndarray) -> float:
        return float(
            (1.0 - ax) * (1.0 - ay) * m[y0, x0]
            + ax * (1.0 - ay) * m[y0, x1]
            + (1.0 - ax) * ay * m[y1, x0]
            + ax * ay * m[y1, x1]
        )

    return _bilinear(map_x), _bilinear(map_y)


def remap_det_left_rectified_to_distorted(
    det: dict,
    rectifier: StereoRectifier,
) -> dict:
    """校正图 infer 的 det_left → 畸变原图像素（用于原图 preview 叠图）。"""
    out = dict(det)
    kp = np.asarray(det.get("keypoints"), dtype=np.float32).copy()
    for i in range(kp.shape[0]):
        if kp[i, 0] <= 0:
            continue
        x, y = sample_rectified_to_distorted_pixel(
            rectifier.map1x,
            rectifier.map1y,
            float(kp[i, 0]),
            float(kp[i, 1]),
        )
        kp[i, 0] = x
        kp[i, 1] = y
    out["keypoints"] = kp
    box = det.get("box_xyxy")
    if box is not None and len(box) >= 4:
        corners = ((box[0], box[1]), (box[2], box[3]))
        xs: list[float] = []
        ys: list[float] = []
        for u, v in corners:
            x, y = sample_rectified_to_distorted_pixel(
                rectifier.map1x, rectifier.map1y, float(u), float(v)
            )
            xs.append(x)
            ys.append(y)
        out["box_xyxy"] = (
            int(min(xs)),
            int(min(ys)),
            int(max(xs)),
            int(max(ys)),
        )
    return out


def projection_matrices_rectified(calib: StereoCalibration) -> tuple[np.ndarray, np.ndarray]:
    """stereoRectify 的 P1/P2（3×4），输入为校正图像素坐标。"""
    p1 = np.asarray(calib.P1, dtype=np.float64)
    p2 = np.asarray(calib.P2, dtype=np.float64)
    return p1, p2


class GpuStereoRectifyMaps:
    """OpenCV remap 表上传 GPU（供 NV12 remap+letterbox CUDA kernel）。"""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        map1x_ptr: int,
        map1y_ptr: int,
        map2x_ptr: int,
        map2y_ptr: int,
        _bufs: list,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.map1x_ptr = int(map1x_ptr)
        self.map1y_ptr = int(map1y_ptr)
        self.map2x_ptr = int(map2x_ptr)
        self.map2y_ptr = int(map2y_ptr)
        self._bufs = _bufs

    @classmethod
    def from_rectifier(cls, rectifier: StereoRectifier) -> GpuStereoRectifyMaps:
        from platform.jetson.cuda_runtime import CudaBuffer

        bufs: list[CudaBuffer] = []
        ptrs: list[int] = []
        for arr in (rectifier.map1x, rectifier.map1y, rectifier.map2x, rectifier.map2y):
            flat = np.ascontiguousarray(arr, dtype=np.float32)
            buf = CudaBuffer(flat.nbytes)
            buf.copy_from_host(flat)
            bufs.append(buf)
            ptrs.append(buf.ptr)
        return cls(
            width=rectifier.width,
            height=rectifier.height,
            map1x_ptr=ptrs[0],
            map1y_ptr=ptrs[1],
            map2x_ptr=ptrs[2],
            map2y_ptr=ptrs[3],
            _bufs=bufs,
        )

    def left_maps(self) -> tuple[int, int]:
        return self.map1x_ptr, self.map1y_ptr

    def right_maps(self) -> tuple[int, int]:
        return self.map2x_ptr, self.map2y_ptr

    def close(self) -> None:
        for buf in self._bufs:
            buf.free()
        self._bufs = []
