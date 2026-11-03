"""实时队列：满则丢旧、消费取最新（latest-wins）。"""

from __future__ import annotations

import queue
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")

ReleaseFn = Callable[[Any], None]


def _noop_release(_: Any) -> None:
    return None


def try_take_latest_from_queue(
    q: queue.Queue,
    *,
    release: ReleaseFn | None = None,
) -> tuple[Any, int] | None:
    """非阻塞：prep_q 无料返回 None；有料则排空队列只保留最新。"""
    rel = release or _noop_release
    try:
        item = q.get_nowait()
    except queue.Empty:
        return None
    dropped = 0
    while True:
        try:
            old = item
            item = q.get_nowait()
            rel(old)
            dropped += 1
        except queue.Empty:
            break
    return item, dropped


def take_latest_from_queue(
    q: queue.Queue,
    timeout: float,
    *,
    release: ReleaseFn | None = None,
) -> tuple[Any, int] | None:
    """阻塞取一帧，再排空队列只保留最新；返回 (item, dropped_count)。"""
    rel = release or _noop_release
    try:
        item = q.get(timeout=timeout)
    except queue.Empty:
        return None
    dropped = 0
    while True:
        try:
            old = item
            item = q.get_nowait()
            rel(old)
            dropped += 1
        except queue.Empty:
            break
    return item, dropped


def put_latest_to_queue(
    q: queue.Queue,
    item: Any,
    *,
    release: ReleaseFn | None = None,
    put_timeout: float = 0.5,
) -> int:
    """写入队列；满则丢最旧直到能放入。返回丢弃数量。"""
    rel = release or _noop_release
    dropped = 0
    while not q.full():
        try:
            q.put(item, timeout=put_timeout)
            return dropped
        except queue.Full:
            pass
        try:
            old = q.get_nowait()
            rel(old)
            dropped += 1
        except queue.Empty:
            continue
    try:
        old = q.get_nowait()
        rel(old)
        dropped += 1
    except queue.Empty:
        pass
    try:
        q.put_nowait(item)
    except queue.Full:
        rel(item)
        dropped += 1
    return dropped
