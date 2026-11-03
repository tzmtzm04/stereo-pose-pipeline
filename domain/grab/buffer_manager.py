"""NVMM 帧引用调度（对标 ZED SDK 内部 buffer 生命周期）。"""

from __future__ import annotations

import threading
from typing import Callable, Optional


class BufferLease:
    """单持有者 lease；release 后从 BufferManager 注销。"""

    __slots__ = ("_manager", "_tag", "_released")

    def __init__(self, manager: "BufferManager", tag: str) -> None:
        self._manager = manager
        self._tag = tag
        self._released = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self._manager._on_lease_released(self._tag)


class BufferManager:
    """
    多消费者共享同一 NVMM 帧：全部 lease 释放后调用 on_final_release。

    R0 默认录制走 gst copy，主要给后续 ref 零拷贝路径预留。
    """

    def __init__(self, on_final_release: Callable[[], None]) -> None:
        self._on_final_release = on_final_release
        self._holders: set[str] = set()
        self._lock = threading.Lock()
        self._finalized = False

    def acquire(self, tag: str) -> BufferLease:
        with self._lock:
            if self._finalized:
                raise RuntimeError(f"BufferManager finalized, cannot acquire {tag!r}")
            self._holders.add(str(tag))
        return BufferLease(self, str(tag))

    def force_release(self) -> None:
        with self._lock:
            if self._finalized:
                return
            self._holders.clear()
            self._finalized = True
        self._on_final_release()

    def _on_lease_released(self, tag: str) -> None:
        finalize = False
        with self._lock:
            self._holders.discard(tag)
            if not self._holders and not self._finalized:
                self._finalized = True
                finalize = True
        if finalize:
            self._on_final_release()

    @property
    def pending_holders(self) -> frozenset[str]:
        with self._lock:
            return frozenset(self._holders)
