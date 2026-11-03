"""Shutdown 辅助：有界等待，避免 teardown 永久阻塞。"""

from __future__ import annotations

import threading
from typing import Callable


def run_with_timeout(
    label: str,
    fn: Callable[[], None],
    *,
    timeout_sec: float = 5.0,
) -> bool:
    """在后台线程执行 fn，主线程最多等 timeout_sec。返回是否在时限内完成。"""
    done = threading.Event()
    err: list[BaseException] = []

    def _worker() -> None:
        try:
            fn()
        except BaseException as exc:
            err.append(exc)
        finally:
            done.set()

    th = threading.Thread(target=_worker, name=f"shutdown_{label}", daemon=True)
    th.start()
    ok = done.wait(timeout=max(0.5, float(timeout_sec)))
    if not ok:
        print(
            f"WARN: shutdown {label} 超时 ({timeout_sec:.0f}s)，继续后续步骤",
            flush=True,
        )
        return False
    if err:
        print(f"WARN: shutdown {label} 异常: {err[0]}", flush=True)
        return False
    return True
