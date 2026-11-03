"""实时黑盒句柄注册表（SDK 私有；orchestration 经 sdk.realtime 绑定）。"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from app.sdk.jpeg_slot import LatestJpegSlot
    from core.display.pose_store import PoseStore
    from core.display.video_ring import DisplayVideoRing
    from platform.jetson.nv12_gl_display import Nv12GlDisplay

_registry: RealtimeRegistry | None = None


class RealtimeRegistry:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pose_store: PoseStore | None = None
        self._video_ring: DisplayVideoRing | None = None
        self._gl: Nv12GlDisplay | None = None
        self._jpeg_slot: LatestJpegSlot | None = None
        self._stop: threading.Event | None = None
        self._quit: threading.Event | None = None
        # 语义：本场录制已封盘、可开下一场（相机常开时 fd 仍可持有）
        self._camera_released: threading.Event | None = None
        self._end_recording: Callable[[], None] | None = None
        self._recording_sid: str | None = None
        self._engine_alive = False

    def bind_pose_store(self, store: PoseStore) -> None:
        with self._lock:
            self._pose_store = store

    def bind_pipeline_control(
        self,
        stop: threading.Event,
        quit_evt: threading.Event,
    ) -> None:
        with self._lock:
            self._stop = stop
            self._quit = quit_evt

    def bind_camera_released(self, evt: threading.Event) -> None:
        """快停解阻塞：录制 idle / 可开下一场（非必须已关 video fd）。"""
        with self._lock:
            self._camera_released = evt

    def bind_end_recording(self, fn: Callable[[], None] | None) -> None:
        with self._lock:
            self._end_recording = fn

    def set_recording_sid(self, sid: str | None) -> None:
        with self._lock:
            self._recording_sid = sid

    def recording_sid(self) -> str | None:
        with self._lock:
            return self._recording_sid

    def set_engine_alive(self, alive: bool) -> None:
        with self._lock:
            self._engine_alive = bool(alive)

    def engine_alive(self) -> bool:
        with self._lock:
            return self._engine_alive

    def bind_video_ring(self, ring: DisplayVideoRing) -> None:
        with self._lock:
            self._video_ring = ring

    def bind_gl_display(self, gl: Nv12GlDisplay) -> None:
        with self._lock:
            self._gl = gl

    def pose_store(self) -> PoseStore | None:
        with self._lock:
            return self._pose_store

    def video_ring(self) -> DisplayVideoRing | None:
        with self._lock:
            return self._video_ring

    def gl_display(self) -> Nv12GlDisplay | None:
        with self._lock:
            return self._gl

    def bind_jpeg_slot(self, slot: LatestJpegSlot) -> None:
        with self._lock:
            self._jpeg_slot = slot

    def jpeg_slot(self) -> LatestJpegSlot | None:
        with self._lock:
            return self._jpeg_slot

    def request_stop(self) -> None:
        """结束本场录制（相机常开）；不拆 pipeline。"""
        with self._lock:
            fn = self._end_recording
        if fn is not None:
            try:
                fn()
                return
            except Exception as exc:
                print(f"WARN: end_recording 失败: {exc}", flush=True)
        # 非 persistent / 未绑定：回退为停 pipeline（CLI 路径）
        with self._lock:
            if self._quit is not None:
                self._quit.set()
            if self._stop is not None:
                self._stop.set()

    def request_shutdown(self) -> None:
        """停引擎 + 释放相机。"""
        with self._lock:
            if self._quit is not None:
                self._quit.set()
            if self._stop is not None:
                self._stop.set()

    def wait_camera_released(self, timeout: float = 12.0) -> bool:
        with self._lock:
            evt = self._camera_released
        if evt is None:
            return False
        return bool(evt.wait(timeout=max(0.1, float(timeout))))

    def clear_session_bindings(self) -> None:
        """清本场 handle 侧绑定；引擎仍活时保留 pose/jpeg/pipeline。"""
        with self._lock:
            if self._engine_alive:
                return
            self._pose_store = None
            self._video_ring = None
            self._gl = None
            self._jpeg_slot = None
            self._stop = None
            self._quit = None
            self._camera_released = None
            self._end_recording = None
            self._recording_sid = None

    def clear(self) -> None:
        """全清（仅引擎已停或进程退出时）。"""
        with self._lock:
            if self._engine_alive:
                # 兼容旧调用：引擎存活时不拆共享资源
                return
            self._pose_store = None
            self._video_ring = None
            self._gl = None
            self._jpeg_slot = None
            self._stop = None
            self._quit = None
            self._camera_released = None
            self._end_recording = None
            self._recording_sid = None


def get_realtime_registry() -> RealtimeRegistry:
    global _registry
    if _registry is None:
        _registry = RealtimeRegistry()
    return _registry
