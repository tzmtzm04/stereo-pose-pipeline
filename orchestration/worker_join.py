"""Pipeline worker join：cap.release 前必须等 pull/infer 线程退出。"""

from __future__ import annotations

import threading
import time

DEFAULT_WORKER_JOIN_TIMEOUT_SEC = 15.0
_JOIN_SLICE_SEC = 0.5


def join_pipeline_workers(
    threads: list[tuple[str, threading.Thread]],
    *,
    stop: threading.Event | None = None,
    timeout_sec: float = DEFAULT_WORKER_JOIN_TIMEOUT_SEC,
) -> list[str]:
    """Join workers until dead or timeout. Returns labels still alive."""
    if not threads:
        return []
    if stop is not None:
        stop.set()
    deadline = time.monotonic() + max(1.0, float(timeout_sec))
    still_alive: list[str] = []
    joined: list[str] = []
    for label, th in threads:
        if not th.is_alive():
            joined.append(label)
            continue
        while th.is_alive() and time.monotonic() < deadline:
            if stop is not None:
                stop.set()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            th.join(timeout=min(_JOIN_SLICE_SEC, remaining))
        if th.is_alive():
            still_alive.append(label)
            print(
                f"WARN: pipeline worker 未在 {timeout_sec:.0f}s 内退出: "
                f"{label} ({th.name})",
                flush=True,
            )
        else:
            joined.append(label)
    if joined:
        print(f"pipeline workers 已退出: {', '.join(joined)}", flush=True)
    return still_alive
