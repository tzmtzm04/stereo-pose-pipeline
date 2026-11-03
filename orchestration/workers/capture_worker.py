"""采集线程：StereoGrabRuntime.grab() → capture_q。"""

from __future__ import annotations

import queue
import threading
import time
from typing import Any

from app.route2.session_config import Route2SessionConfig
from platform.jetson.capture import JetsonGstNvmmCapture


def run_capture_worker(
    cfg: Route2SessionConfig,
    cap: JetsonGstNvmmCapture,
    capture_q: queue.Queue,
    stop: threading.Event,
    *,
    grab_runtime: Any | None = None,
) -> None:
    fail_streak = 0
    fail_max = max(5, int(cfg.capture_fail_max))
    if grab_runtime is None:
        raise RuntimeError("capture worker 需要 StereoGrabRuntime")
    while not stop.is_set():
        t0 = time.perf_counter()
        ok, grab_frame = grab_runtime.grab()
        pull_ms = (time.perf_counter() - t0) * 1000.0
        if not ok or grab_frame is None:
            fail_streak += 1
            if fail_streak == 1 or fail_streak % 10 == 0:
                print(
                    f"读帧超时/失败 streak={fail_streak}/{fail_max} "
                    f"(last wait {pull_ms:.0f}ms)，重试…",
                    flush=True,
                )
            if fail_streak >= fail_max:
                print(
                    f"读帧连续失败 {fail_streak} 次，采集退出",
                    flush=True,
                )
                stop.set()
                break
            continue
        fail_streak = 0
        if grab_frame.prepared is not None:
            try:
                item = grab_frame.consume_prepared()
            except Exception:
                grab_frame.release()
                continue
        elif grab_frame.sbs is not None:
            try:
                sbs, pull_ms = grab_frame.consume_sbs()
                item = (sbs, pull_ms)
            except Exception:
                grab_frame.release()
                continue
        else:
            grab_frame.release()
            continue
        grab_frame.release()
        try:
            capture_q.put(item, timeout=0.5)
        except queue.Full:
            from orchestration.queue_util import put_latest_to_queue

            put_latest_to_queue(
                capture_q,
                item,
                release=_release_capture_item,
                put_timeout=0.0,
            )
            continue


def _release_capture_item(item: object) -> None:
    if isinstance(item, tuple) and item:
        _release_sbs(item[0])
    elif hasattr(item, "release"):
        try:
            item.release()
        except Exception:
            pass


def _release_sbs(sbs: Any) -> None:
    try:
        sbs.release()
    except Exception:
        pass
