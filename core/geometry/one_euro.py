"""One Euro 2D 滤波。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


class OneEuroFilter:
    def __init__(
        self,
        t0: float,
        x0: float,
        dx0: float = 0.0,
        min_cutoff: float = 1.0,
        beta: float = 0.0,
        d_cutoff: float = 1.0,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = float(x0)
        self.dx_prev = float(dx0)
        self.t_prev = float(t0)

    def smoothing_factor(self, t_e: float, cutoff: float) -> float:
        r = 2 * math.pi * cutoff * t_e
        return r / (r + 1)

    def exponential_smoothing(self, a: float, x: float, x_prev: float) -> float:
        return a * x + (1 - a) * x_prev

    def __call__(self, t: float, x: float) -> float:
        t_e = t - self.t_prev
        if t_e <= 1e-5:
            return x
        a_d = self.smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = self.exponential_smoothing(a_d, dx, self.dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self.smoothing_factor(t_e, cutoff)
        x_hat = self.exponential_smoothing(a, x, self.x_prev)
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class Keypoint2DOneEuroFilter:
    def __init__(
        self,
        *,
        person_id: int = 0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        teleport_px: float = 500.0,
    ) -> None:
        self.person_id = int(person_id)
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.teleport_px = float(teleport_px)
        self._filters: dict[tuple[int, int, int], OneEuroFilter] = {}
        self._last_timestamp = -1.0

    def reset(self) -> None:
        self._filters = {}
        self._last_timestamp = -1.0

    def filter(self, timestamp: float, keypoints: np.ndarray) -> np.ndarray:
        kpts = np.asarray(keypoints, dtype=np.float32)
        if self._last_timestamp > timestamp:
            self.reset()
        self._last_timestamp = timestamp
        out = kpts.copy()
        for i in range(out.shape[0]):
            x, y = float(kpts[i, 0]), float(kpts[i, 1])
            out[i, 0] = self._filter_coord(timestamp, i, 0, x)
            out[i, 1] = self._filter_coord(timestamp, i, 1, y)
        return out

    def _filter_coord(
        self,
        timestamp: float,
        joint: int,
        coord: int,
        value: float,
    ) -> float:
        if math.isnan(value):
            return value
        uid = (self.person_id, joint, coord)
        if uid not in self._filters:
            self._filters[uid] = OneEuroFilter(
                timestamp,
                value,
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                d_cutoff=self.d_cutoff,
            )
            return value
        filt = self._filters[uid]
        prev = filt.x_prev
        if math.isnan(prev) or abs(value - prev) > self.teleport_px:
            self._filters[uid] = OneEuroFilter(
                timestamp,
                value,
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                d_cutoff=self.d_cutoff,
            )
            return value
        return filt(timestamp, value)


class Keypoint3DOneEuroFilter:
    """三角化后的 3D 关键点 One Euro（Pose2Sim [filtering.one_euro] 思路）。"""

    def __init__(
        self,
        *,
        min_cutoff: float = 4.0,
        beta: float = 1.5,
        d_cutoff: float = 1.0,
        max_miss_frames: int = 6,
        teleport_m: float = 0.5,
    ) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.max_miss_frames = int(max_miss_frames)
        self.teleport_m = float(teleport_m)
        self._filters: dict[tuple[int, int], OneEuroFilter] = {}
        self._last_timestamp = -1.0
        self._last_kpts: np.ndarray | None = None
        self._miss = 0

    def reset(self) -> None:
        self._filters = {}
        self._last_timestamp = -1.0
        self._last_kpts = None
        self._miss = 0

    def filter(self, timestamp: float, kpts3d: np.ndarray | None) -> np.ndarray | None:
        if kpts3d is None:
            self._miss += 1
            if self._last_kpts is not None and self._miss <= self.max_miss_frames:
                return self._last_kpts.copy()
            return None

        if self._last_timestamp > timestamp:
            self.reset()

        self._last_timestamp = timestamp
        k = np.asarray(kpts3d, dtype=np.float32).copy()
        out = k.copy()

        for ji in range(k.shape[0]):
            conf = float(k[ji, 3]) if k.shape[1] > 3 else 1.0
            if conf <= 0.0:
                continue
            for coord in range(3):
                val = float(k[ji, coord])
                if not np.isfinite(val):
                    continue
                out[ji, coord] = self._filter_coord(timestamp, ji, coord, val)

        self._last_kpts = out.copy()
        self._miss = 0
        return out

    def _filter_coord(
        self,
        timestamp: float,
        joint: int,
        coord: int,
        value: float,
    ) -> float:
        uid = (joint, coord)
        if uid not in self._filters:
            self._filters[uid] = OneEuroFilter(
                timestamp,
                value,
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                d_cutoff=self.d_cutoff,
            )
            return value
        filt = self._filters[uid]
        prev = filt.x_prev
        if not np.isfinite(prev) or abs(value - prev) > self.teleport_m:
            self._filters[uid] = OneEuroFilter(
                timestamp,
                value,
                min_cutoff=self.min_cutoff,
                beta=self.beta,
                d_cutoff=self.d_cutoff,
            )
            return value
        return filt(timestamp, value)


class StereoPose2DOneEuroFilter:
    """左右眼 2D 检测滤波，三角化前调用。"""

    def __init__(
        self,
        *,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        d_cutoff: float = 1.0,
        max_miss_frames: int = 6,
        teleport_px: float = 500.0,
    ) -> None:
        self.max_miss_frames = int(max_miss_frames)
        self._left = Keypoint2DOneEuroFilter(
            person_id=0,
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
            teleport_px=teleport_px,
        )
        self._right = Keypoint2DOneEuroFilter(
            person_id=1,
            min_cutoff=min_cutoff,
            beta=beta,
            d_cutoff=d_cutoff,
            teleport_px=teleport_px,
        )
        self._last_left: dict | None = None
        self._last_right: dict | None = None
        self._miss_left = 0
        self._miss_right = 0

    def reset(self) -> None:
        self._left.reset()
        self._right.reset()
        self._last_left = None
        self._last_right = None
        self._miss_left = 0
        self._miss_right = 0

    def _filter_eye(
        self,
        filt: Keypoint2DOneEuroFilter,
        det: dict | None,
        timestamp: float,
        last: dict | None,
        miss: int,
    ) -> tuple[dict | None, dict | None, int]:
        if det is None:
            miss += 1
            if last is not None and miss <= self.max_miss_frames:
                return last, last, miss
            return None, None, miss
        kpts = filt.filter(timestamp, det["keypoints"])
        out = dict(det)
        out["keypoints"] = kpts
        return out, out, 0

    def filter_pair(
        self,
        timestamp: float,
        det_left: dict | None,
        det_right: dict | None,
    ) -> tuple[dict | None, dict | None]:
        fl, self._last_left, self._miss_left = self._filter_eye(
            self._left, det_left, timestamp, self._last_left, self._miss_left
        )
        fr, self._last_right, self._miss_right = self._filter_eye(
            self._right, det_right, timestamp, self._last_right, self._miss_right
        )
        return fl, fr


@dataclass(frozen=True)
class OneEuro2DConfig:
    """离线 C×N×J 轨迹 One Euro（与 Route2 实时默认同量级）。"""

    enabled: bool = True
    min_cutoff: float = 1.0
    beta: float = 0.007
    d_cutoff: float = 1.0
    fps: float = 60.0
    teleport_px: float = 500.0


def filter_pose2d_stereo_one_euro(
    points: np.ndarray,
    scores: np.ndarray,
    cfg: OneEuro2DConfig,
) -> np.ndarray:
    """C×N×J×2 逐相机、逐帧 One Euro（用帧序号作时间轴）。"""
    pts = np.asarray(points, dtype=np.float64).copy()
    sc = np.asarray(scores, dtype=np.float64)
    n_cams, n_frames, n_joints, _ = pts.shape
    dt = 1.0 / max(float(cfg.fps), 1e-6)

    for ci in range(n_cams):
        filt = Keypoint2DOneEuroFilter(
            person_id=ci,
            min_cutoff=float(cfg.min_cutoff),
            beta=float(cfg.beta),
            d_cutoff=float(cfg.d_cutoff),
            teleport_px=float(cfg.teleport_px),
        )
        for fi in range(n_frames):
            t = fi * dt
            kpts = np.full((n_joints, 3), np.nan, dtype=np.float32)
            for ji in range(n_joints):
                xy = pts[ci, fi, ji]
                if not np.all(np.isfinite(xy)):
                    continue
                kpts[ji, 0] = float(xy[0])
                kpts[ji, 1] = float(xy[1])
                if sc.ndim >= 3:
                    kpts[ji, 2] = float(sc[ci, fi, ji])
                else:
                    kpts[ji, 2] = 1.0
            kpts_f = filt.filter(t, kpts)
            for ji in range(n_joints):
                if np.isfinite(kpts_f[ji, 0]) and np.isfinite(kpts_f[ji, 1]):
                    pts[ci, fi, ji] = kpts_f[ji, :2]
    return pts
