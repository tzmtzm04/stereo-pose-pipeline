"""长寿命实时引擎：相机常开，按 session 切录（≈ ZED open 一次 + enable/disable Recording）。"""

from __future__ import annotations

import threading
import traceback
from typing import Callable

from app.route2.session_config import Route2SessionConfig
from app.sdk._registry import get_realtime_registry
from app.sdk.types import PreviewConfig
from orchestration.session import Route2LiveSession

_engine: LiveEngine | None = None
_engine_lock = threading.Lock()


class LiveEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._session: Route2LiveSession | None = None
        self._error: BaseException | None = None
        self._boot_done = threading.Event()
        self._cfg_builder: Callable[[str, PreviewConfig], Route2SessionConfig] | None = None
        self._preview: PreviewConfig | None = None
        self._first_sid: str | None = None

    @property
    def thread(self) -> threading.Thread | None:
        return self._thread

    @property
    def error(self) -> BaseException | None:
        return self._error

    def is_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def is_recording(self, session_id: str | None = None) -> bool:
        sess = self._session
        if sess is None:
            return False
        return sess.is_recording(session_id)

    def wait_boot(self, timeout: float = 60.0) -> bool:
        return bool(self._boot_done.wait(timeout=max(0.5, float(timeout))))

    def start_recording(
        self,
        session_id: str,
        *,
        preview: PreviewConfig,
        cfg_builder: Callable[[str, PreviewConfig], Route2SessionConfig],
    ) -> None:
        """首次 boot+开录；之后只切 session 目录。"""
        with self._lock:
            self._cfg_builder = cfg_builder
            self._preview = preview
            if self._thread is None or not self._thread.is_alive():
                self._error = None
                self._boot_done.clear()
                self._first_sid = str(session_id)
                self._thread = threading.Thread(
                    target=self._run_engine,
                    name=f"sdk_live_engine",
                    daemon=True,
                )
                self._thread.start()
            else:
                sess = self._session
                if sess is None or not self._boot_done.is_set():
                    raise RuntimeError("live engine 尚未就绪")
                if sess.is_recording(session_id):
                    return
                if sess.recording:
                    sess.end_recording()
                sess.begin_recording(str(session_id))
                return

        if not self.wait_boot(timeout=90.0):
            raise RuntimeError("live engine boot 超时")
        if self._error is not None:
            raise RuntimeError(f"live engine boot 失败: {self._error}") from self._error

    def end_recording(self) -> None:
        sess = self._session
        if sess is not None:
            sess.end_recording()

    def wait_ready_for_next(self, timeout: float = 12.0) -> bool:
        return get_realtime_registry().wait_camera_released(timeout=timeout)

    def close(self, *, join_timeout: float = 60.0) -> None:
        """停引擎并释放相机。"""
        from platform.jetson.gst_capture import (
            device_held_by_current_process,
            force_close_device_fds,
        )

        with self._lock:
            get_realtime_registry().request_shutdown()
            th = self._thread
            device = "/dev/video0"
            sess = self._session
            if sess is not None and sess.cfg is not None:
                device = str(getattr(sess.cfg, "device", device) or device)
        if th is not None and th.is_alive():
            self.end_recording()
            get_realtime_registry().request_shutdown()
            th.join(timeout=max(1.0, float(join_timeout)))
        if th is not None and th.is_alive():
            print(
                f"WARN: live engine 线程未在 {join_timeout:.0f}s 内退出，强制关 video fd",
                flush=True,
            )
        if device_held_by_current_process(device):
            force_close_device_fds(device)
        with self._lock:
            self._session = None
            self._thread = None
            self._boot_done.clear()
        reg = get_realtime_registry()
        reg.set_engine_alive(False)
        reg.bind_end_recording(None)
        # 引擎已停，允许全清
        reg.set_engine_alive(False)
        # bypass clear() guard
        with reg._lock:
            reg._engine_alive = False
        reg.clear()

    def _run_engine(self) -> None:
        assert self._cfg_builder is not None
        assert self._preview is not None
        assert self._first_sid is not None
        try:
            cfg = self._cfg_builder(self._first_sid, self._preview)
            sess = Route2LiveSession(cfg, persistent=True)
            self._session = sess

            def _mark_booted() -> None:
                # run() 内 _booted.set()；此处轮询或 hook
                pass

            # monkey: wait until session._booted inside run — poll from outside after start
            boot_watch = threading.Thread(
                target=self._watch_boot,
                args=(sess,),
                name="live_engine_boot_watch",
                daemon=True,
            )
            boot_watch.start()
            sess.run()
        except BaseException as exc:
            self._error = exc
            print(f"live engine 异常: {exc}", flush=True)
            traceback.print_exc()
            self._boot_done.set()
        finally:
            self._session = None
            get_realtime_registry().set_engine_alive(False)
            get_realtime_registry().bind_end_recording(None)

    def _watch_boot(self, sess: Route2LiveSession) -> None:
        deadline_ok = sess._booted.wait(timeout=90.0)
        if deadline_ok or self._error is not None:
            self._boot_done.set()
        else:
            self._error = self._error or TimeoutError("boot wait")
            self._boot_done.set()


def get_live_engine() -> LiveEngine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = LiveEngine()
        return _engine


def reset_live_engine_for_tests() -> None:
    global _engine
    with _engine_lock:
        if _engine is not None:
            try:
                _engine.close(join_timeout=30.0)
            except Exception:
                pass
        _engine = None
