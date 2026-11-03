"""L1 三角化 3D 历史（仅内存）。"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import numpy as np


@dataclass(frozen=True)
class Pose3DSample:
    capture_seq: int
    infer_seq: int
    kpts3d: Optional[np.ndarray]
    tri_ok: bool
    det_left: Optional[dict] = None
    det_right: Optional[dict] = None
    opencap20_kpts3d: Optional[np.ndarray] = None
    soma_joints78: Optional[np.ndarray] = None
    body_loss: Optional[float] = None
    filter_2d: Optional[str] = None


class L1PoseHistory:
    def __init__(self, *, history_max_samples: int = 2000) -> None:
        self._history_max_samples = max(1, history_max_samples)
        self._history: Deque[Pose3DSample] = deque(maxlen=self._history_max_samples)
        self._lock = threading.Lock()

    @property
    def history_capacity(self) -> int:
        return self._history_max_samples

    def append(
        self,
        capture_seq: int,
        infer_seq: int,
        kpts3d: np.ndarray | None,
    ) -> Pose3DSample:
        tri_ok = kpts3d is not None
        kpts_copy: Optional[np.ndarray] = None
        if kpts3d is not None:
            kpts_copy = np.asarray(kpts3d, dtype=np.float32).copy()
        sample = Pose3DSample(
            capture_seq=int(capture_seq),
            infer_seq=int(infer_seq),
            kpts3d=kpts_copy,
            tri_ok=tri_ok,
        )
        with self._lock:
            self._history.append(sample)
        return sample

    def history_len(self) -> int:
        with self._lock:
            return len(self._history)

    def latest_sample(self) -> Optional[Pose3DSample]:
        with self._lock:
            if not self._history:
                return None
            return self._history[-1]

    def snapshot(self) -> List[Pose3DSample]:
        with self._lock:
            return list(self._history)
