"""3D 关键点卡尔曼平滑（髋部锚定 + 相对坐标 + 输出 EMA）。

对齐 BodPoesNet/src/stereo/kalman_pose3d.py：三角化后逐帧调用。
"""

from __future__ import annotations

import numpy as np

NUM_JOINTS = 17
_HIP_LEFT, _HIP_RIGHT = 11, 12


def measure_z_scale(
    z_m: float,
    *,
    z_ref: float,
    z_alpha: float,
) -> float:
    """左相机深度 Z（米）→ 观测噪声缩放：Z 越大越不信任观测（滤得更重）。"""
    if float(z_alpha) <= 0.0:
        return 1.0
    z = max(abs(float(z_m)), 0.05)
    ref = max(float(z_ref), 0.05)
    return max(0.25, 1.0 + float(z_alpha) * (z / ref - 1.0))


class JointKalman3D:
    """状态 [x,y,z,vx,vy,vz]，观测 [x,y,z]（米）。"""

    def __init__(
        self,
        dt: float,
        process_std: float,
        measure_std: float,
        *,
        measure_z_ref: float = 2.5,
        measure_z_alpha: float = 0.5,
    ) -> None:
        self.dt = float(dt)
        self._measure_std = float(measure_std)
        self._z_ref = float(measure_z_ref)
        self._z_alpha = float(measure_z_alpha)
        self.initialized = False
        self.conf = 0.0
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 10.0
        dt = self.dt
        self.F = np.array(
            [
                [1, 0, 0, dt, 0, 0],
                [0, 1, 0, 0, dt, 0],
                [0, 0, 1, 0, 0, dt],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        q = float(process_std) ** 2
        self.Q = np.eye(6, dtype=np.float64) * q
        self.Q[3:, 3:] *= 2.0

    def _measurement_R(self, conf: float, z_depth: float | None = None) -> np.ndarray:
        c = max(float(conf), 0.2)
        sigma = self._measure_std / c
        if z_depth is not None:
            sigma *= measure_z_scale(
                z_depth, z_ref=self._z_ref, z_alpha=self._z_alpha
            )
        r = sigma**2
        return np.eye(3, dtype=np.float64) * r

    def predict(self) -> None:
        if not self.initialized:
            return
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(
        self,
        z: np.ndarray,
        conf: float,
        *,
        z_depth: float | None = None,
    ) -> None:
        z = np.asarray(z, dtype=np.float64).reshape(3)
        if not self.initialized:
            self.x[:3] = z
            self.x[3:] = 0.0
            self.P = np.eye(6, dtype=np.float64) * 0.01
            self.initialized = True
            self.conf = float(conf)
            return
        z_for_scale = float(z_depth) if z_depth is not None else float(z[2])
        R = self._measurement_R(conf, z_for_scale)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float64)
        self.P = (I - K @ self.H) @ self.P
        self.conf = max(self.conf * 0.92, float(conf))

    @property
    def position(self) -> np.ndarray:
        return self.x[:3].astype(np.float32)


def _hip_from_measurement(k: np.ndarray, min_conf: float) -> np.ndarray | None:
    pts = []
    if k[_HIP_LEFT, 3] >= min_conf and k[_HIP_LEFT, 2] > 0.01:
        pts.append(k[_HIP_LEFT, :3])
    if k[_HIP_RIGHT, 3] >= min_conf and k[_HIP_RIGHT, 2] > 0.01:
        pts.append(k[_HIP_RIGHT, :3])
    if not pts:
        return None
    return np.mean(pts, axis=0).astype(np.float64)


class Pose3DKalmanSmoother:
    """
    髋部绝对坐标 + 关节相对髋部坐标分别滤波，再合成世界坐标。

    输入/输出：(17, 4) — x,y,z,conf（米，左相机系）。
    """

    def __init__(
        self,
        *,
        fps: float = 30.0,
        process_std: float = 0.002,
        measure_std: float = 0.045,
        max_miss_frames: int = 15,
        output_ema: float = 0.32,
        measure_z_ref: float = 2.5,
        measure_z_alpha: float = 0.5,
    ) -> None:
        dt = 1.0 / max(float(fps), 1.0)
        self.max_miss_frames = int(max_miss_frames)
        self.output_ema = float(np.clip(output_ema, 0.0, 1.0))
        kz = dict(measure_z_ref=measure_z_ref, measure_z_alpha=measure_z_alpha)
        self._hip = JointKalman3D(
            dt, process_std * 0.5, measure_std * 0.8, **kz
        )
        self._joints = [
            JointKalman3D(dt, process_std, measure_std, **kz)
            for _ in range(NUM_JOINTS)
        ]
        self._miss_frames = 0
        self._has_track = False
        self._prev_out: np.ndarray | None = None

    def reset(self) -> None:
        self._miss_frames = 0
        self._has_track = False
        self._prev_out = None
        self._hip.initialized = False
        self._hip.conf = 0.0
        self._hip.x[:] = 0.0
        self._hip.P = np.eye(6, dtype=np.float64) * 10.0
        for f in self._joints:
            f.initialized = False
            f.conf = 0.0
            f.x[:] = 0.0
            f.P = np.eye(6, dtype=np.float64) * 10.0

    def _compose_output(self) -> np.ndarray:
        hip = self._hip.position.astype(np.float64)
        out = np.zeros((NUM_JOINTS, 4), dtype=np.float32)
        for i, f in enumerate(self._joints):
            if f.initialized:
                out[i, :3] = (f.position.astype(np.float64) + hip).astype(np.float32)
                out[i, 3] = min(f.conf, self._hip.conf)
        return out

    def _apply_output_ema(self, out: np.ndarray) -> np.ndarray:
        if self.output_ema <= 0.0 or self._prev_out is None:
            self._prev_out = out.copy()
            return out
        a = self.output_ema
        blended = out.copy()
        mask = (out[:, 3] > 0) & (self._prev_out[:, 3] > 0)
        blended[mask, :3] = a * out[mask, :3] + (1.0 - a) * self._prev_out[mask, :3]
        self._prev_out = blended.copy()
        return blended

    def filter(
        self,
        kpts3d: np.ndarray | None,
        *,
        min_conf: float,
    ) -> np.ndarray | None:
        self._hip.predict()
        for f in self._joints:
            f.predict()

        any_meas = False
        hip_meas: np.ndarray | None = None
        if kpts3d is not None:
            k = np.asarray(kpts3d, dtype=np.float32)
            if k.shape[0] >= NUM_JOINTS:
                hip_meas = _hip_from_measurement(k, min_conf)
                if hip_meas is not None:
                    confs = []
                    if k[_HIP_LEFT, 3] >= min_conf and k[_HIP_LEFT, 2] > 0.01:
                        confs.append(float(k[_HIP_LEFT, 3]))
                    if k[_HIP_RIGHT, 3] >= min_conf and k[_HIP_RIGHT, 2] > 0.01:
                        confs.append(float(k[_HIP_RIGHT, 3]))
                    hip_conf = min(confs) if confs else 0.5
                    self._hip.update(
                        hip_meas, hip_conf, z_depth=float(hip_meas[2])
                    )
                    any_meas = True
                for i in range(NUM_JOINTS):
                    if k[i, 3] >= min_conf and k[i, 2] > 0.01:
                        rel = k[i, :3].astype(np.float64)
                        if hip_meas is not None:
                            rel = rel - hip_meas
                        self._joints[i].update(
                            rel,
                            float(k[i, 3]),
                            z_depth=float(k[i, 2]),
                        )
                        any_meas = True

        if any_meas:
            self._miss_frames = 0
            self._has_track = True
        elif self._has_track and self._miss_frames < self.max_miss_frames:
            self._miss_frames += 1
        else:
            self.reset()
            return None

        if not self._has_track or not self._hip.initialized:
            return None

        out = self._compose_output()
        return self._apply_output_ema(out)


class Keypoints3DKalmanFilter:
    """逐关节 3D 卡尔曼（支持 WholeBody 133 点），三角化后调用。"""

    def __init__(
        self,
        *,
        fps: float = 60.0,
        process_std: float = 0.002,
        measure_std: float = 0.045,
        max_miss_frames: int = 15,
        output_ema: float = 0.32,
        measure_z_ref: float = 2.5,
        measure_z_alpha: float = 0.5,
    ) -> None:
        dt = 1.0 / max(float(fps), 1.0)
        self._dt = dt
        self._process_std = float(process_std)
        self._measure_std = float(measure_std)
        self._z_ref = float(measure_z_ref)
        self._z_alpha = float(measure_z_alpha)
        self.max_miss_frames = int(max_miss_frames)
        self.output_ema = float(np.clip(output_ema, 0.0, 1.0))
        self._joints: list[JointKalman3D] = []
        self._miss_frames = 0
        self._has_track = False
        self._prev_out: np.ndarray | None = None

    def reset(self) -> None:
        self._miss_frames = 0
        self._has_track = False
        self._prev_out = None
        self._joints.clear()

    def _ensure_joint(self, idx: int) -> JointKalman3D:
        while len(self._joints) <= idx:
            self._joints.append(
                JointKalman3D(
                    self._dt,
                    self._process_std,
                    self._measure_std,
                    measure_z_ref=self._z_ref,
                    measure_z_alpha=self._z_alpha,
                )
            )
        return self._joints[idx]

    def _apply_output_ema(self, out: np.ndarray) -> np.ndarray:
        if self.output_ema <= 0.0 or self._prev_out is None:
            self._prev_out = out.copy()
            return out
        a = self.output_ema
        blended = out.copy()
        mask = (out[:, 3] > 0) & (self._prev_out[:, 3] > 0)
        blended[mask, :3] = a * out[mask, :3] + (1.0 - a) * self._prev_out[mask, :3]
        self._prev_out = blended.copy()
        return blended

    def filter(
        self,
        kpts3d: np.ndarray | None,
        *,
        min_conf: float,
    ) -> np.ndarray | None:
        if kpts3d is None:
            for f in self._joints:
                f.predict()
            if self._has_track and self._miss_frames < self.max_miss_frames:
                self._miss_frames += 1
            else:
                self.reset()
                return None
        else:
            k = np.asarray(kpts3d, dtype=np.float32)
            if k.ndim != 2 or k.shape[1] < 4:
                return None
            j_count = int(k.shape[0])
            any_meas = False
            for i in range(j_count):
                f = self._ensure_joint(i)
                f.predict()
                if (
                    float(k[i, 3]) >= float(min_conf)
                    and float(np.linalg.norm(k[i, :3])) > 1e-4
                ):
                    f.update(k[i, :3], float(k[i, 3]))
                    any_meas = True

            if any_meas:
                self._miss_frames = 0
                self._has_track = True
            elif self._has_track and self._miss_frames < self.max_miss_frames:
                self._miss_frames += 1
            else:
                self.reset()
                return None

            if not self._has_track:
                return None

            out = k.copy()
            for i in range(j_count):
                f = self._joints[i]
                if f.initialized:
                    out[i, :3] = f.position
                    out[i, 3] = max(float(out[i, 3]), float(f.conf))
            return self._apply_output_ema(out)

        if not self._has_track or not self._joints:
            return None
        # predict-only 帧：用上一帧结构输出预测位置
        if self._prev_out is None:
            return None
        out = self._prev_out.copy()
        for i, f in enumerate(self._joints):
            if f.initialized and i < out.shape[0]:
                out[i, :3] = f.position
        return self._apply_output_ema(out)
