"""Display 出图线 → UI 层契约（策略 D + P4 EGL/GL 或 OpenCV BGR）。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.display.pose_store import PoseDisplaySample


@dataclass
class DisplayTick:
    """Producer 一帧视频 + capture_seq（pose 由 UI 侧 PoseStore exact match）。"""

    capture_seq: int
    bgr: np.ndarray | None = None
    nv12: Any | None = None
    nv12_sbs: bool = False
    pose: PoseDisplaySample | None = None
    pull_ms: float = 0.0
    preview_ms: float = 0.0
    wait_ms: float = 0.0


def _release_tick_nv12(tick: DisplayTick | None) -> None:
    if tick is None or tick.nv12 is None:
        return
    try:
        tick.nv12.release()
    except Exception:
        pass
    tick.nv12 = None


class DisplayFrameSlot:
    """Latest-frame 双缓冲：producer 持续覆写，UI 非阻塞取最新（不积压队列）。"""

    def __init__(self, *, depth: int = 2) -> None:
        self._depth = max(2, int(depth))
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._version = 0
        self._closed = False
        self._tick: DisplayTick | None = None
        self._bufs: list[np.ndarray | None] = [None] * self._depth
        self._write_idx = 0
        self._prod_ok = 0
        self._prod_pull_fails = 0

    def record_pull_fail(self) -> int:
        with self._lock:
            self._prod_pull_fails += 1
            return int(self._prod_pull_fails)

    def prod_stats(self) -> tuple[int, int]:
        with self._lock:
            return int(self._prod_ok), int(self._prod_pull_fails)

    @property
    def version(self) -> int:
        with self._lock:
            return int(self._version)

    def acquire_write_buffer(self, shape: tuple[int, ...], dtype: np.dtype = np.uint8) -> np.ndarray | None:
        """Producer 取下一写入缓冲（双缓冲之一），供 GPU D2H 直写。"""
        with self._lock:
            if self._closed:
                return None
            buf = self._bufs[self._write_idx]
            if buf is None or buf.shape != shape or buf.dtype != dtype:
                buf = np.empty(shape, dtype=dtype)
                self._bufs[self._write_idx] = buf
            return buf

    def publish_buffer(
        self,
        bgr: np.ndarray,
        *,
        capture_seq: int,
        pull_ms: float,
        preview_ms: float,
        wait_ms: float = 0.0,
        pose: PoseDisplaySample | None = None,
    ) -> None:
        """发布 acquire 缓冲（零 copyto）；外源数组则回退 copyto。"""
        with self._lock:
            if self._closed:
                return
            slot_buf = self._bufs[self._write_idx]
            if slot_buf is not None and bgr is slot_buf:
                buf = slot_buf
            else:
                if (
                    slot_buf is None
                    or slot_buf.shape != bgr.shape
                    or slot_buf.dtype != bgr.dtype
                ):
                    slot_buf = np.ascontiguousarray(bgr)
                    self._bufs[self._write_idx] = slot_buf
                else:
                    np.copyto(slot_buf, bgr)
                buf = slot_buf
            self._tick = DisplayTick(
                capture_seq=int(capture_seq),
                bgr=buf,
                pose=pose,
                pull_ms=float(pull_ms),
                preview_ms=float(preview_ms),
                wait_ms=float(wait_ms),
            )
            self._write_idx = (self._write_idx + 1) % self._depth
            self._prod_ok += 1
            self._version += 1
            self._cv.notify_all()

    def publish_tick(self, tick: DisplayTick) -> None:
        """发布一帧；NV12 fd 路径零 BGR 拷贝，覆写时释放旧 nv12 keepalive。"""
        with self._lock:
            if self._closed:
                _release_tick_nv12(tick)
                return
            _release_tick_nv12(self._tick)
            if tick.nv12 is not None:
                self._tick = DisplayTick(
                    capture_seq=int(tick.capture_seq),
                    bgr=None,
                    nv12=tick.nv12,
                    nv12_sbs=bool(tick.nv12_sbs),
                    pose=tick.pose,
                    pull_ms=float(tick.pull_ms),
                    preview_ms=float(tick.preview_ms),
                    wait_ms=float(tick.wait_ms),
                )
            elif tick.bgr is not None:
                slot_buf = self._bufs[self._write_idx]
                src = tick.bgr
                if slot_buf is not None and src is slot_buf:
                    buf = slot_buf
                else:
                    if (
                        slot_buf is None
                        or slot_buf.shape != src.shape
                        or slot_buf.dtype != src.dtype
                    ):
                        slot_buf = np.ascontiguousarray(src)
                        self._bufs[self._write_idx] = slot_buf
                    else:
                        np.copyto(slot_buf, src)
                    buf = slot_buf
                self._tick = DisplayTick(
                    capture_seq=int(tick.capture_seq),
                    bgr=buf,
                    pose=tick.pose,
                    pull_ms=float(tick.pull_ms),
                    preview_ms=float(tick.preview_ms),
                    wait_ms=float(tick.wait_ms),
                )
                self._write_idx = (self._write_idx + 1) % self._depth
            else:
                return
            self._prod_ok += 1
            self._version += 1
            self._cv.notify_all()

    def publish(self, tick: DisplayTick) -> None:
        """覆写最新帧；满槽语义：只保留最后一帧。"""
        self.publish_tick(tick)

    def _snapshot_if_newer(self, since_version: int) -> tuple[int, DisplayTick | None]:
        """调用方须已持有 self._lock。"""
        if self._version <= int(since_version) or self._tick is None:
            return int(since_version), None
        t = self._tick
        bgr = t.bgr
        if bgr is not None:
            bgr = np.ascontiguousarray(bgr).copy()
        return int(self._version), DisplayTick(
            capture_seq=int(t.capture_seq),
            bgr=bgr,
            nv12=t.nv12,
            nv12_sbs=bool(t.nv12_sbs),
            pose=t.pose,
            pull_ms=float(t.pull_ms),
            preview_ms=float(t.preview_ms),
            wait_ms=float(t.wait_ms),
        )

    def take_latest(self, since_version: int) -> tuple[int, DisplayTick | None]:
        """若自 since_version 以来有新帧，返回 (version, tick)；否则 (since_version, None)。"""
        with self._lock:
            return self._snapshot_if_newer(since_version)

    def wait_latest(
        self,
        since_version: int,
        *,
        timeout: float,
    ) -> tuple[int, DisplayTick | None]:
        import time

        end = time.monotonic() + max(0.0, float(timeout))
        with self._cv:
            while True:
                if self._closed:
                    return int(self._version), None
                ver, tick = self._snapshot_if_newer(since_version)
                if tick is not None:
                    return ver, tick
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return int(since_version), None
                self._cv.wait(timeout=remaining)

    def close(self) -> None:
        with self._lock:
            _release_tick_nv12(self._tick)
            self._tick = None
            self._closed = True
            self._cv.notify_all()


def acquire_latest_for_ui(
    slot: DisplayFrameSlot,
    since_version: int,
    *,
    wait_timeout: float = 0.008,
) -> tuple[int, DisplayTick | None, float]:
    """ZED 语义：非阻塞取最新；无新帧则短等；合并中间帧只渲染最后一帧。"""
    import time

    t_wait = time.perf_counter()
    version, tick = slot.take_latest(since_version)
    if tick is None:
        version, tick = slot.wait_latest(since_version, timeout=wait_timeout)
    slot_wait_ms = (time.perf_counter() - t_wait) * 1000.0
    if tick is None:
        return since_version, None, slot_wait_ms
    while True:
        v2, t2 = slot.take_latest(version)
        if t2 is None:
            break
        version, tick = v2, t2
    return version, tick, slot_wait_ms
