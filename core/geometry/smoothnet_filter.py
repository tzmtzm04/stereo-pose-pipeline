"""SmoothNet 2D 时序滤波（替换 One Euro，三角化前左右眼独立）。"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from core.geometry.smoothnet_model import SmoothNetNumpy

NUM_JOINTS = 17
POSE_DIM_2D = NUM_JOINTS * 2


class Keypoint2DSmoothNetFilter:
    """单眼 COCO-17 2D 滑动窗口 SmoothNet。"""

    def __init__(
        self,
        model: SmoothNetNumpy,
        *,
        person_id: int = 0,
        max_miss_frames: int = 6,
    ) -> None:
        self._model = model
        self.person_id = int(person_id)
        self.max_miss_frames = int(max_miss_frames)
        self._window = int(model.cfg.window_size)
        self._buf: deque[np.ndarray] = deque(maxlen=self._window)
        self._miss = 0
        self._last_out: np.ndarray | None = None

    def reset(self) -> None:
        self._buf.clear()
        self._miss = 0
        self._last_out = None

    def filter(self, keypoints: np.ndarray) -> np.ndarray:
        kpts = np.asarray(keypoints, dtype=np.float32)
        xy = kpts[:, :2].copy()
        self._buf.append(xy)
        self._miss = 0

        if len(self._buf) < self._window:
            self._last_out = kpts.copy()
            return kpts

        seq = np.stack(list(self._buf), axis=0).astype(np.float32)  # (T, 17, 2)
        inp = seq.transpose(1, 2, 0).reshape(1, POSE_DIM_2D, self._window)
        out = self._model.forward(inp)  # (1, 34, output_size)
        smoothed_xy = out[0, :, -1].reshape(NUM_JOINTS, 2)
        result = kpts.copy()
        result[:, :2] = smoothed_xy
        self._last_out = result
        return result

    def hold_last(self) -> np.ndarray | None:
        self._miss += 1
        if self._last_out is not None and self._miss <= self.max_miss_frames:
            return self._last_out.copy()
        return None


class StereoPose2DSmoothNetFilter:
    """左右眼 2D 检测 SmoothNet 滤波，接口与 StereoPose2DOneEuroFilter 一致。"""

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        max_miss_frames: int = 6,
    ) -> None:
        self._model = SmoothNetNumpy.load_npz(checkpoint)
        if self._model.cfg.pose_dim != POSE_DIM_2D:
            raise ValueError(
                f"SmoothNet pose_dim={self._model.cfg.pose_dim}，"
                f"期望 COCO-17 2D dim={POSE_DIM_2D}"
            )
        self._left = Keypoint2DSmoothNetFilter(
            self._model, person_id=0, max_miss_frames=max_miss_frames
        )
        self._right = Keypoint2DSmoothNetFilter(
            self._model, person_id=1, max_miss_frames=max_miss_frames
        )
        self._last_left: dict | None = None
        self._last_right: dict | None = None
        print(
            f"SmoothNet 2D: window={self._model.cfg.window_size} "
            f"output={self._model.cfg.output_size} "
            f"checkpoint={Path(checkpoint).name}",
            flush=True,
        )

    @property
    def window_size(self) -> int:
        return int(self._model.cfg.window_size)

    def reset(self) -> None:
        self._left.reset()
        self._right.reset()
        self._last_left = None
        self._last_right = None

    def _filter_eye(
        self,
        filt: Keypoint2DSmoothNetFilter,
        det: dict | None,
        last: dict | None,
    ) -> tuple[dict | None, dict | None]:
        if det is None:
            held = filt.hold_last()
            if held is not None and last is not None:
                out = dict(last)
                out["keypoints"] = held
                return out, out
            return None, None
        kpts = filt.filter(det["keypoints"])
        out = dict(det)
        out["keypoints"] = kpts
        return out, out

    def filter_pair(
        self,
        timestamp: float,
        det_left: dict | None,
        det_right: dict | None,
    ) -> tuple[dict | None, dict | None]:
        _ = timestamp  # SmoothNet 按帧序，不用连续时间戳
        fl, self._last_left = self._filter_eye(self._left, det_left, self._last_left)
        fr, self._last_right = self._filter_eye(
            self._right, det_right, self._last_right
        )
        return fl, fr
