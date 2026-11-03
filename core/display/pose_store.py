"""推理 pose 按 capture_seq 缓存；display 侧 match 查表。"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Optional

import numpy as np

from contracts.seq_util import int_attr
from core.infer.pose3d_pipeline import StereoPose3DResult

MatchPolicy = Literal["latest_le", "exact"]


@dataclass(frozen=True)
class PoseDisplaySample:
    capture_seq: int
    infer_seq: int
    det_left: dict | None
    kpts3d: np.ndarray | None
    tri_ok: bool
    det_right: dict | None = None
    infer_parallel_ms: float = 0.0
    capture_ms: float = 0.0
    infer_egl_ms: float = 0.0
    infer_lb_ms: float = 0.0
    infer_trt_ms: float = 0.0
    infer_post_ms: float = 0.0
    infer_drain_ms: float = 0.0
    infer_wait_ms: float = 0.0
    infer_unmap_ms: float = 0.0
    infer_trt_enqueue_ms: float = 0.0
    infer_trt_d2h_sync_ms: float = 0.0
    tri_ms: float = 0.0


class PoseStore:
    """线程安全；infer 写入，display match 读取。"""

    def __init__(self, *, max_samples: int = 3000) -> None:
        self._max = max(64, int(max_samples))
        self._history: Deque[PoseDisplaySample] = deque(maxlen=self._max)
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._latest_capture_seq = -1

    def put_from_pose(self, pose: StereoPose3DResult) -> PoseDisplaySample:
        kpts = pose.kpts3d
        kpts_copy: np.ndarray | None = None
        if kpts is not None:
            kpts_copy = np.asarray(kpts, dtype=np.float32).copy()
        def _copy_det(det: dict | None) -> dict | None:
            if det is None:
                return None
            out = dict(det)
            kp = det.get("keypoints")
            if kp is not None:
                out["keypoints"] = np.asarray(kp, dtype=np.float32).copy()
            return out

        det_copy = _copy_det(pose.det_left)
        det_r_copy = _copy_det(pose.det_right)
        bd = pose.infer.par_breakdown()
        sample = PoseDisplaySample(
            capture_seq=int_attr(pose.infer, "capture_seq", default=-1),
            infer_seq=int(pose.infer.frame_id),
            det_left=det_copy,
            det_right=det_r_copy,
            kpts3d=kpts_copy,
            tri_ok=bool(pose.tri_ok),
            infer_parallel_ms=float(pose.infer.infer_parallel_ms),
            capture_ms=float(pose.infer.capture_ms),
            infer_egl_ms=float(bd.egl_ms),
            infer_lb_ms=float(bd.lb_ms),
            infer_trt_ms=float(bd.trt_ms),
            infer_post_ms=float(bd.post_ms),
            infer_drain_ms=float(bd.drain_ms),
            infer_wait_ms=float(bd.wait_ms),
            infer_unmap_ms=float(bd.unmap_ms),
            infer_trt_enqueue_ms=float(bd.trt_enqueue_ms),
            infer_trt_d2h_sync_ms=float(bd.trt_d2h_sync_ms),
            tri_ms=float(getattr(pose, "tri_ms", 0.0)),
        )
        with self._lock:
            self._history.append(sample)
            self._latest_capture_seq = max(
                self._latest_capture_seq,
                int(sample.capture_seq),
            )
            self._cv.notify_all()
        return sample

    @property
    def latest_capture_seq(self) -> int:
        with self._lock:
            return int(self._latest_capture_seq)

    def _next_pose_after_unlocked(self, after_seq: int) -> Optional[PoseDisplaySample]:
        """调用方须已持有 self._lock；返回 seq > after_seq 的最小 capture_seq。"""
        target: PoseDisplaySample | None = None
        for item in self._history:
            seq = int(item.capture_seq)
            if seq <= int(after_seq):
                continue
            if target is None or seq < int(target.capture_seq):
                target = item
        return target

    def next_pose_after(self, after_seq: int) -> Optional[PoseDisplaySample]:
        """非阻塞：下一帧待显示 pose（最小 capture_seq > after_seq）。"""
        with self._lock:
            return self._next_pose_after_unlocked(after_seq)

    def wait_for_pose_after(
        self,
        after_seq: int,
        *,
        timeout: float = 0.5,
    ) -> Optional[PoseDisplaySample]:
        """阻塞直到出现 capture_seq > after_seq 的 pose（display 等骨架）。"""
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cv:
            while True:
                pose = self._next_pose_after_unlocked(after_seq)
                if pose is not None:
                    return pose
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._cv.wait(timeout=remaining)

    def wait_for_next_pose_after(
        self,
        after_seq: int,
        *,
        timeout: float = 0.5,
    ) -> Optional[PoseDisplaySample]:
        """按序等下一 pose（与 wait_for_pose_after 相同，语义更明确）。"""
        return self.wait_for_pose_after(after_seq, timeout=timeout)

    def match(
        self,
        capture_seq: int,
        *,
        policy: MatchPolicy = "latest_le",
    ) -> Optional[PoseDisplaySample]:
        seq = int(capture_seq)
        with self._lock:
            if policy == "exact":
                for item in reversed(self._history):
                    if item.capture_seq == seq:
                        return item
                return None
            for item in reversed(self._history):
                if item.capture_seq <= seq:
                    return item
            return None

    def __len__(self) -> int:
        with self._lock:
            return len(self._history)
