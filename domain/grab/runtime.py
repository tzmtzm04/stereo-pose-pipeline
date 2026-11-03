"""StereoGrabRuntime：统一 grab 语义（≈ sl::Camera::grab）。"""

from __future__ import annotations

import time
from typing import Any

from contracts.seq_util import int_attr
from domain.grab.grab_frame import GrabFrame
from platform.jetson.nv12_frame import Nv12SbsFrame


class StereoGrabRuntime:
    """
    应用层 grab 入口（≈ sl::Camera::grab）。

    grab 成功后的扇出（capture_seq 为时间轴）：

        grab() → capture_seq++
            ├─ infer 队列 → TRT → PoseStore + session_pose.jsonl
            ├─ mjpeg ingress submit → session.mjpeg（非阻塞）
            └─ display NV12 tee 支路 → DisplayVideoRing（不等 infer）
    """

    def __init__(
        self,
        cap: Any,
    ) -> None:
        self._cap = cap

    def grab(self) -> tuple[bool, GrabFrame | None]:
        t0 = time.perf_counter()
        ok, sbs = self._cap.read_nv12()
        pull_ms = (time.perf_counter() - t0) * 1000.0
        if not ok or sbs is None:
            return False, None
        if not isinstance(sbs, Nv12SbsFrame):
            try:
                sbs.release()
            except Exception:
                pass
            return False, None

        capture_seq = int_attr(sbs, "capture_seq", default=-1)
        return True, GrabFrame(
            capture_seq=capture_seq,
            pull_ms=pull_ms,
            recording_submitted=False,
            prepared=None,
            sbs=sbs,
        )
