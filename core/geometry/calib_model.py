"""双目标定数据结构。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class StereoCalibration:
    image_width: int
    image_height: int
    pattern_cols: int
    pattern_rows: int
    square_size_m: float
    K1: np.ndarray
    D1: np.ndarray
    K2: np.ndarray
    D2: np.ndarray
    R: np.ndarray
    T: np.ndarray
    R1: np.ndarray
    R2: np.ndarray
    P1: np.ndarray
    P2: np.ndarray
    Q: np.ndarray
    reprojection_error: float = 0.0
    num_pairs: int = 0
    notes: str = ""

    @property
    def baseline_m(self) -> float:
        return float(np.linalg.norm(self.T))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StereoCalibration:
        return cls(
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            pattern_cols=int(data["pattern_cols"]),
            pattern_rows=int(data["pattern_rows"]),
            square_size_m=float(data["square_size_m"]),
            K1=np.asarray(data["K1"], dtype=np.float64),
            D1=np.asarray(data["D1"], dtype=np.float64).ravel(),
            K2=np.asarray(data["K2"], dtype=np.float64),
            D2=np.asarray(data["D2"], dtype=np.float64).ravel(),
            R=np.asarray(data["R"], dtype=np.float64),
            T=np.asarray(data["T"], dtype=np.float64).reshape(3, 1),
            R1=np.asarray(data["R1"], dtype=np.float64),
            R2=np.asarray(data["R2"], dtype=np.float64),
            P1=np.asarray(data["P1"], dtype=np.float64),
            P2=np.asarray(data["P2"], dtype=np.float64),
            Q=np.asarray(data["Q"], dtype=np.float64),
            reprojection_error=float(data.get("reprojection_error", 0.0)),
            num_pairs=int(data.get("num_pairs", 0)),
            notes=str(data.get("notes", "")),
        )
