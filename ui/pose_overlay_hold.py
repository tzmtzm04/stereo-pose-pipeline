"""UI 叠图保持：首检前可原画；首检后无骨架则冻结上一帧叠图。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from core.display.pose_store import PoseDisplaySample
from ui.pose_driven_ui import PoseDrivenStep, StepState

if TYPE_CHECKING:
    from core.display.display_tick import DisplayTick


@dataclass
class PoseOverlayHoldState:
    """首检后缓存最后一帧「原画+骨架」；GL 路径另保 tick+pose 供重绘。"""

    armed: bool = False
    composite: np.ndarray | None = None
    status: str = ""
    capture_seq: int = -1
    pose: PoseDisplaySample | None = None
    gl_tick: DisplayTick | None = None

    def save_skeleton_frame(
        self,
        composite: np.ndarray,
        *,
        status: str,
        pose: PoseDisplaySample,
        capture_seq: int,
    ) -> None:
        self.armed = True
        self.composite = np.ascontiguousarray(composite)
        self.status = str(status)
        self.pose = pose
        self.capture_seq = int(capture_seq)

    def save_gl_skeleton(
        self,
        tick: DisplayTick,
        *,
        pose: PoseDisplaySample,
    ) -> None:
        from core.display.video_ring import release_display_tick

        if self.gl_tick is not None:
            release_display_tick(self.gl_tick)
        self.gl_tick = tick
        self.pose = pose
        self.capture_seq = int(tick.capture_seq)
        self.armed = True

    def release_gl_tick(self) -> None:
        from core.display.video_ring import release_display_tick

        if self.gl_tick is not None:
            release_display_tick(self.gl_tick)
            self.gl_tick = None

    def has_hold(self) -> bool:
        return self.armed and self.composite is not None

    def has_gl_hold(self) -> bool:
        return (
            self.armed
            and self.gl_tick is not None
            and self.gl_tick.nv12 is not None
            and self.pose is not None
        )


_HOLD_WHEN_ARMED: frozenset[StepState] = frozenset(
    {
        "hold_overlay",
        "wait_pose",
    }
)


def is_skeleton_render_step(step: PoseDrivenStep) -> bool:
    """当前步可叠骨架（含回到画面后的新检测）。"""
    if step.state != "render" or step.tick is None or step.resolved is None:
        return False
    return step.resolved.pose is not None


def should_freeze_pose_overlay(step: PoseDrivenStep, hold: PoseOverlayHoldState) -> bool:
    """首检后：当前步无法叠骨架 → 复用上一帧合成。"""
    if is_skeleton_render_step(step):
        return False
    if not hold.armed:
        return False
    if step.state == "render":
        return True
    return step.state in _HOLD_WHEN_ARMED
