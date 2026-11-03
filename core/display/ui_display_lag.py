"""UI 层播放滞后：视频与骨架同 seq，整体比 live 晚 N 帧（不动 infer/display/rec）。"""

from __future__ import annotations

import time
from collections import deque

from app.route2.session_config import Route2SessionConfig
from core.display.display_tick import DisplayTick, _release_tick_nv12
from core.display.live_compose import (
    ResolvedDisplayPose,
    resolve_exact_display_pose,
)
from core.display.pose_store import PoseStore

LAG_POSE_POLL_SEC = 0.002


class UiLagBuffer:
    """UI 侧 FIFO：攒满 lag 帧后，每次输出最旧一帧（与 PoseStore exact 对齐）。"""

    def __init__(self, lag_frames: int) -> None:
        self._lag = max(0, int(lag_frames))
        self._pending: deque[DisplayTick] = deque()
        self._ready_logged = False

    @property
    def lag_frames(self) -> int:
        return self._lag

    def push(self, tick: DisplayTick) -> None:
        self._pending.append(tick)
        cap = max(self._lag + 8, 16) if self._lag > 0 else 2
        while len(self._pending) > cap:
            dropped = self._pending.popleft()
            _release_tick_nv12(dropped)

    def fill_count(self) -> int:
        return len(self._pending)

    def peek_ready(self) -> DisplayTick | None:
        if self._lag <= 0:
            if self._pending:
                return self._pending[0]
            return None
        if len(self._pending) > self._lag:
            return self._pending[0]
        return None

    def pop_ready(self) -> DisplayTick | None:
        ready = self.peek_ready()
        if ready is None:
            return None
        return self._pending.popleft()

    def note_first_display(self, capture_seq: int) -> None:
        if self._ready_logged or self._lag <= 0:
            return
        self._ready_logged = True
        print(
            f"ui lag 播放就绪 lag={self._lag} 首帧 seq={capture_seq}",
            flush=True,
        )


def resolve_lagged_pose(
    cfg: Route2SessionConfig,
    store: PoseStore,
    capture_seq: int,
) -> ResolvedDisplayPose:
    """滞后帧只用 exact match，保证画面与骨架同一 capture_seq。"""
    return resolve_exact_display_pose(cfg, store, capture_seq)


def wait_lagged_pose(
    cfg: Route2SessionConfig,
    store: PoseStore,
    capture_seq: int,
    *,
    extra_wait_ms: float | None = None,
) -> ResolvedDisplayPose:
    """短等滞后帧 pose；超时仍返回（由调用方出队，避免缓冲卡死）。"""
    wait_ms = (
        float(extra_wait_ms)
        if extra_wait_ms is not None
        else max(33.0, float(cfg.display_pose_ui_wait_ms) * 1000.0 * 2.0)
    )
    resolved = resolve_lagged_pose(cfg, store, capture_seq)
    if resolved.pose is not None:
        return resolved
    deadline = time.monotonic() + max(0.0, wait_ms) / 1000.0
    while time.monotonic() < deadline:
        resolved = resolve_lagged_pose(cfg, store, capture_seq)
        if resolved.pose is not None:
            return resolved
        time.sleep(LAG_POSE_POLL_SEC)
    return resolved
