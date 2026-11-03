"""UI 滞后缓冲 + pose 解析（gl / imshow 共用）。"""

from __future__ import annotations

from typing import Literal

from app.route2.session_config import Route2SessionConfig
from core.display.display_tick import DisplayTick
from core.display.live_compose import DisplayPoseResolver, ResolvedDisplayPose
from core.display.pose_store import PoseStore
from core.display.ui_display_lag import (
    UiLagBuffer,
    wait_lagged_pose,
)

EmitState = Literal["render", "filling", "skip"]

_pose_miss_log = 0
_POSE_MISS_LOG_EVERY = 60


def step_lagged_ui_frame(
    cfg: Route2SessionConfig,
    store: PoseStore,
    lag_buf: UiLagBuffer,
    pose_resolver: DisplayPoseResolver,
    tick: DisplayTick,
    *,
    live_capture_seq: int,
) -> tuple[EmitState, DisplayTick | None, ResolvedDisplayPose | None, int]:
    """处理 live tick：滞后缓冲出队；无检测时仍显示原画。"""
    global _pose_miss_log
    live_seq = int(live_capture_seq)
    ready = lag_buf.peek_ready()
    if ready is not None:
        capture_seq = int(ready.capture_seq)
        if lag_buf.lag_frames > 0:
            resolved = wait_lagged_pose(cfg, store, capture_seq)
        else:
            resolved = pose_resolver.resolve(store, capture_seq)
        if resolved.pose is None:
            _pose_miss_log += 1
            if _pose_miss_log == 1 or _pose_miss_log % _POSE_MISS_LOG_EVERY == 0:
                print(
                    f"ui lag 无检测 seq={capture_seq} "
                    f"(累计={_pose_miss_log})，显示原画",
                    flush=True,
                )
            out = lag_buf.pop_ready()
            if out is None:
                lag_buf.push(tick)
                return "filling", None, None, live_seq
            lag_buf.push(tick)
            lag_buf.note_first_display(capture_seq)
            return "render", out, ResolvedDisplayPose(pose=None), live_seq
        out = lag_buf.pop_ready()
        if out is None:
            lag_buf.push(tick)
            return "filling", None, None, live_seq
        lag_buf.push(tick)
        lag_buf.note_first_display(capture_seq)
        return "render", out, resolved, live_seq

    lag_buf.push(tick)
    return "filling", None, None, live_seq


def lag_buffer_status(cfg: Route2SessionConfig, lag_buf: UiLagBuffer) -> str:
    lag = int(cfg.display_ui_lag_frames)
    if lag <= 0:
        return ""
    n = lag_buf.fill_count()
    return f"UI 缓冲 {n}/{lag}…  q quit"
