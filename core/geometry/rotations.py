"""轴角 / 旋转矩阵工具（文档 Step 1 / FK）。"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros(3, dtype=np.float64)
    return (v / n).astype(np.float64)


def rodrigues(axis_angle: np.ndarray) -> np.ndarray:
    aa = np.asarray(axis_angle, dtype=np.float64).reshape(3)
    return Rotation.from_rotvec(aa).as_matrix()


def axis_angle_from_matrix(rot: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(np.asarray(rot, dtype=np.float64)).as_rotvec().astype(
        np.float32
    )


def rotation_between(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    a = normalize(v_from)
    b = normalize(v_to)
    if float(np.linalg.norm(a)) < 1e-8 or float(np.linalg.norm(b)) < 1e-8:
        return np.eye(3, dtype=np.float64)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 0.9999:
        return np.eye(3, dtype=np.float64)
    if dot < -0.9999:
        ortho = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(a[0]) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        axis = normalize(np.cross(a, ortho))
        return rodrigues(axis * np.pi)
    cross = np.cross(a, b)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + skew + skew @ skew * (1.0 / (1.0 + dot))
