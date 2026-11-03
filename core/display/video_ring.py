"""按 capture_seq 缓存 display 视频，供 UI 骨架驱动原画。"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Deque

import numpy as np

from core.display.display_tick import DisplayTick, _release_tick_nv12

DEFAULT_DISPLAY_VIDEO_RING_FRAMES = 32
# NV12 fd 须尽快 release；略加深环，避免 latest_pose 原画被挤掉导致 UI 回落旧帧
DEFAULT_DISPLAY_VIDEO_RING_FRAMES_NV12 = 12


def release_display_tick(tick: DisplayTick | None) -> None:
    """UI 消费后释放 NV12 keepalive。"""
    _release_tick_nv12(tick)


def _release_stored_tick(tick: DisplayTick | None) -> None:
    if tick is None:
        return
    _release_tick_nv12(tick)


def _store_tick(tick: DisplayTick) -> DisplayTick:
    """入环拷贝：BGR 深拷贝，NV12 fd 转移所有权（调用方勿再 release）。"""
    if tick.nv12 is not None:
        return DisplayTick(
            capture_seq=int(tick.capture_seq),
            nv12=tick.nv12,
            nv12_sbs=bool(tick.nv12_sbs),
            pull_ms=float(tick.pull_ms),
            preview_ms=float(tick.preview_ms),
            wait_ms=float(tick.wait_ms),
        )
    if tick.bgr is not None:
        return DisplayTick(
            capture_seq=int(tick.capture_seq),
            bgr=np.ascontiguousarray(tick.bgr).copy(),
            pull_ms=float(tick.pull_ms),
            preview_ms=float(tick.preview_ms),
            wait_ms=float(tick.wait_ms),
        )
    return DisplayTick(
        capture_seq=int(tick.capture_seq),
        pull_ms=float(tick.pull_ms),
        preview_ms=float(tick.preview_ms),
        wait_ms=float(tick.wait_ms),
    )


class DisplayVideoRing:
    """浅环：producer push，UI 按 capture_seq 取原画；NV12 显示后须 take+release。"""

    def __init__(
        self,
        *,
        max_frames: int = DEFAULT_DISPLAY_VIDEO_RING_FRAMES,
        nv12_fd: bool = False,
    ) -> None:
        self._nv12_fd = bool(nv12_fd)
        cap = (
            DEFAULT_DISPLAY_VIDEO_RING_FRAMES_NV12
            if self._nv12_fd
            else int(max_frames)
        )
        self._max_frames = max(2 if self._nv12_fd else 8, int(cap))
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._by_seq: dict[int, DisplayTick] = {}
        self._order: Deque[int] = deque()
        self._latest_capture_seq = -1
        self._closed = False
        self._prod_ok = 0
        self._prod_pull_fails = 0
        self._dropped = 0

    @property
    def nv12_fd(self) -> bool:
        return self._nv12_fd

    @property
    def max_frames(self) -> int:
        return int(self._max_frames)

    @property
    def latest_capture_seq(self) -> int:
        with self._lock:
            return int(self._latest_capture_seq)

    def len(self) -> int:
        with self._lock:
            return len(self._order)

    def record_pull_fail(self) -> int:
        with self._lock:
            self._prod_pull_fails += 1
            return int(self._prod_pull_fails)

    def prod_stats(self) -> tuple[int, int]:
        with self._lock:
            return int(self._prod_ok), int(self._prod_pull_fails)

    def _pop_unlocked(self, seq: int) -> DisplayTick | None:
        tick = self._by_seq.pop(int(seq), None)
        if tick is None:
            return None
        try:
            self._order.remove(int(seq))
        except ValueError:
            pass
        return tick

    def push(self, tick: DisplayTick) -> None:
        seq = int(tick.capture_seq)
        stored = _store_tick(tick)
        with self._lock:
            if self._closed:
                _release_stored_tick(stored)
                return
            prev = self._by_seq.pop(seq, None)
            if prev is not None:
                try:
                    self._order.remove(seq)
                except ValueError:
                    pass
                _release_stored_tick(prev)
            self._by_seq[seq] = stored
            self._order.append(seq)
            while len(self._order) > self._max_frames:
                old_seq = self._order.popleft()
                evicted = self._by_seq.pop(old_seq, None)
                _release_stored_tick(evicted)
                self._dropped += 1
            self._latest_capture_seq = max(self._latest_capture_seq, seq)
            self._prod_ok += 1
            self._cv.notify_all()

    def get(self, capture_seq: int) -> DisplayTick | None:
        """只读探测（不移除）。"""
        seq = int(capture_seq)
        with self._lock:
            tick = self._by_seq.get(seq)
            if tick is None:
                return None
            return DisplayTick(
                capture_seq=int(tick.capture_seq),
                bgr=tick.bgr,
                nv12=tick.nv12,
                nv12_sbs=bool(tick.nv12_sbs),
                pull_ms=float(tick.pull_ms),
                preview_ms=float(tick.preview_ms),
                wait_ms=float(tick.wait_ms),
            )

    def take(self, capture_seq: int) -> DisplayTick | None:
        """取出并移出环（所有权交给调用方，用后须 release_display_tick）。"""
        seq = int(capture_seq)
        with self._lock:
            return self._pop_unlocked(seq)

    def peek_newest_after(self, after_seq: int) -> DisplayTick | None:
        """只读探测最新 seq>after_seq 的帧（不移除，UI 不得 release）。"""
        with self._lock:
            best = -1
            for seq in self._order:
                if int(seq) > int(after_seq) and int(seq) > best:
                    best = int(seq)
            if best < 0:
                return None
            tick = self._by_seq.get(best)
            if tick is None:
                return None
            return DisplayTick(
                capture_seq=int(tick.capture_seq),
                bgr=tick.bgr,
                nv12=tick.nv12,
                nv12_sbs=bool(tick.nv12_sbs),
                pull_ms=float(tick.pull_ms),
                preview_ms=float(tick.preview_ms),
                wait_ms=float(tick.wait_ms),
            )

    def take_newest_after(self, after_seq: int) -> DisplayTick | None:
        with self._lock:
            best = -1
            for seq in self._order:
                if int(seq) > int(after_seq) and int(seq) > best:
                    best = int(seq)
            if best < 0:
                return None
            return self._pop_unlocked(best)

    def wait_and_take(
        self,
        capture_seq: int,
        *,
        timeout: float,
    ) -> tuple[DisplayTick | None, float]:
        seq = int(capture_seq)
        t0 = time.perf_counter()
        deadline = time.monotonic() + max(0.0, float(timeout))
        with self._cv:
            while True:
                tick = self._pop_unlocked(seq)
                if tick is not None:
                    wait_ms = (time.perf_counter() - t0) * 1000.0
                    return tick, wait_ms
                if self._closed:
                    wait_ms = (time.perf_counter() - t0) * 1000.0
                    return None, wait_ms
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    wait_ms = (time.perf_counter() - t0) * 1000.0
                    return None, wait_ms
                self._cv.wait(timeout=remaining)

    def wait_for_seq(
        self,
        capture_seq: int,
        *,
        timeout: float,
    ) -> tuple[DisplayTick | None, float]:
        tick, wait_ms = self.wait_and_take(capture_seq, timeout=timeout)
        return tick, wait_ms

    def prune_keep_latest(self, keep: int) -> int:
        """释放最老帧，缓解 NV12 NVMM 池压力。返回释放数量。"""
        k = max(1, int(keep))
        released = 0
        with self._lock:
            while len(self._order) > k:
                old_seq = self._order.popleft()
                evicted = self._by_seq.pop(old_seq, None)
                _release_stored_tick(evicted)
                released += 1
                self._dropped += 1
            self._cv.notify_all()
        return released

    def close(self) -> None:
        with self._lock:
            for tick in self._by_seq.values():
                _release_stored_tick(tick)
            self._by_seq.clear()
            self._order.clear()
            self._closed = True
            self._cv.notify_all()
