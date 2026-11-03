"""L3 动作判段占位：模拟 wall-clock 延迟后登记起止帧。"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Sequence, Tuple

from domain.l3.config import MOCK_CLIP_SEGMENTS

if TYPE_CHECKING:
    from domain.l3.runtime import L3Runtime

logger = logging.getLogger(__name__)

MOCK_COMPUTE_DELAY_SEC = 0.5

MockSegment = Tuple[int, int, Optional[str]]


@dataclass(frozen=True)
class MockBoundarySpec:
    start_capture_seq: int
    end_capture_seq: int
    label: Optional[str] = None


def default_mock_segments() -> tuple[MockBoundarySpec, ...]:
    return tuple(
        MockBoundarySpec(start, end, label)
        for start, end, label in _normalize_mock_segments(MOCK_CLIP_SEGMENTS)
    )


def _normalize_mock_segments(
    raw: Sequence[Tuple[int, int] | Tuple[int, int, Optional[str]]],
) -> tuple[MockSegment, ...]:
    out: list[MockSegment] = []
    for i, item in enumerate(raw):
        if len(item) == 2:
            start, end = item  # type: ignore[misc]
            out.append((int(start), int(end), f"mock_{i}"))
        else:
            start, end, label = item  # type: ignore[misc]
            out.append((int(start), int(end), label))
    return tuple(out)


class MockActionDetector:
    """
    见到 capture_seq 到达边界帧后，等待 compute_delay_sec，再登记 marker / segment。
    不阻塞 infer 热路径（Timer 在独立线程触发）。
    """

    def __init__(
        self,
        l3: "L3Runtime",
        *,
        segments: Sequence[MockBoundarySpec] | None = None,
        compute_delay_sec: float = MOCK_COMPUTE_DELAY_SEC,
    ) -> None:
        self._l3 = l3
        self._segments = tuple(segments) if segments is not None else default_mock_segments()
        self._compute_delay_sec = max(0.0, float(compute_delay_sec))
        self._lock = threading.Lock()
        self._armed_start: set[int] = set()
        self._armed_end: set[int] = set()
        self._committed_start: set[int] = set()
        self._timers: list[threading.Timer] = []

    def on_capture_seq(self, capture_seq: int) -> None:
        cap = int(capture_seq)
        for spec in self._segments:
            if cap >= spec.start_capture_seq:
                self._maybe_arm_start(
                    spec.start_capture_seq,
                    spec.end_capture_seq,
                    spec.label,
                )
            if cap >= spec.end_capture_seq:
                self._maybe_arm_end(
                    spec.start_capture_seq,
                    spec.end_capture_seq,
                    spec.label,
                )

    def stop(self) -> None:
        with self._lock:
            for t in self._timers:
                t.cancel()
            self._timers.clear()

    def _maybe_arm_start(
        self,
        start: int,
        end: int,
        label: Optional[str],
    ) -> None:
        with self._lock:
            if start in self._armed_start:
                return
            self._armed_start.add(start)
        self._schedule(self._commit_start, start, end, label)

    def _maybe_arm_end(
        self,
        start: int,
        end: int,
        label: Optional[str],
    ) -> None:
        with self._lock:
            if end in self._armed_end:
                return
            self._armed_end.add(end)
        self._schedule(self._commit_end, start, end, label)

    def _schedule(self, fn, start: int, end: int, label: Optional[str]) -> None:
        timer = threading.Timer(
            self._compute_delay_sec,
            fn,
            args=(start, end, label),
        )
        timer.daemon = True
        with self._lock:
            self._timers.append(timer)
        timer.start()

    def _commit_start(self, start: int, end: int, label: Optional[str]) -> None:
        with self._lock:
            if start in self._committed_start:
                return
            self._committed_start.add(start)
        tag = f"{label}_start" if label else "segment_start"
        idx = self._l3.register_action_marker(start, label=tag, kind="segment_start")
        logger.info(
            "mock L3 起始帧 capture_seq=%d segment %d..%d manifest idx=%d",
            start,
            start,
            end,
            idx,
        )

    def _commit_end(self, start: int, end: int, label: Optional[str]) -> None:
        idx = self._l3.register_action_segment(start, end, label=label)
        logger.info(
            "mock L3 终止帧 segment %d..%d → persist manifest idx=%d",
            start,
            end,
            idx,
        )
