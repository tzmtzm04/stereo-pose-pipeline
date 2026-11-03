"""最新 JPEG 帧槽（SDK 内，供预览参考；无 fall_risk 依赖）。"""

from __future__ import annotations

import threading
import time


class LatestJpegSlot:
    """线程安全的最新 JPEG 帧（HTTP /live.mjpg 等盒外消费）。"""

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._jpeg: bytes | None = None
        self._seq = 0

    def publish(self, jpeg: bytes) -> None:
        with self._cond:
            self._jpeg = jpeg
            self._seq += 1
            self._cond.notify_all()

    def snapshot(self) -> tuple[bytes | None, int]:
        with self._cond:
            return self._jpeg, self._seq

    def wait_new(self, after_seq: int, timeout: float = 1.0) -> tuple[bytes | None, int]:
        deadline = time.monotonic() + max(0.0, timeout)
        with self._cond:
            while self._seq <= after_seq:
                remain = deadline - time.monotonic()
                if remain <= 0:
                    return self._jpeg, self._seq
                self._cond.wait(timeout=remain)
            return self._jpeg, self._seq
