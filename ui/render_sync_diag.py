"""UI 合成时序诊断：原画 seq / 骨架 seq / live / pose_latest（仅日志）。

启用：默认开；`ROUTE2_UI_SYNC_DIAG=0` 关。
周期：`ROUTE2_UI_SYNC_DIAG_EVERY=30`（与 ui log_every 独立，默认 30）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from ui.pose_driven_ui import PoseDrivenStep


def ui_sync_diag_enabled() -> bool:
    return os.environ.get("ROUTE2_UI_SYNC_DIAG", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


@dataclass
class RenderSyncDiag:
    log_every: int = 30
    steps: int = 0
    renders: int = 0
    skipped_video_only: int = 0
    overlay_ok: int = 0
    overlay_drop: int = 0
    hold_show: int = 0
    live_lag_sum: int = 0
    pose_lead_sum: int = 0

    @classmethod
    def from_env(cls) -> RenderSyncDiag:
        every = int(os.environ.get("ROUTE2_UI_SYNC_DIAG_EVERY", "30"))
        return cls(log_every=max(1, every))

    @staticmethod
    def _tick_seq(step: PoseDrivenStep) -> int:
        if step.tick is None:
            return -1
        return int(step.tick.capture_seq)

    @staticmethod
    def _pose_seq(step: PoseDrivenStep) -> int:
        if step.resolved is None or step.resolved.pose is None:
            return -1
        return int(step.resolved.pose.capture_seq)

    def note_step(
        self,
        step: PoseDrivenStep,
        *,
        ui_idx: int,
        pose_latest: int,
        dropped: bool = False,
    ) -> None:
        if not ui_sync_diag_enabled():
            return
        self.steps += 1
        video_seq = self._tick_seq(step)
        pose_seq = self._pose_seq(step)
        live = int(step.live_capture_seq)
        if step.state == "render_video_only" and dropped:
            self.skipped_video_only += 1
            n = self.skipped_video_only
            if n == 1 or n % self.log_every == 0:
                print(
                    f"ui_sync SKIP video_only ui={ui_idx} video={video_seq} "
                    f"live={live} pose_latest={pose_latest} (累计={n})",
                    flush=True,
                )
            return
        if step.state in ("hold_overlay", "wait_pose", "wait_video", "skip_stale"):
            if self.steps == 1 or ui_idx % self.log_every == 0:
                print(
                    f"ui_sync step ui={ui_idx} state={step.state} "
                    f"target={step.target_seq} video={video_seq} pose={pose_seq} "
                    f"live={live} pose_latest={pose_latest}",
                    flush=True,
                )

    def note_render(
        self,
        *,
        ui_idx: int,
        path: str,
        video_seq: int,
        pose_seq: int,
        live_seq: int,
        pose_latest: int,
        overlay: bool,
        held: bool,
    ) -> None:
        if not ui_sync_diag_enabled():
            return
        self.renders += 1
        live_lag = int(live_seq) - int(video_seq)
        pose_lead = int(pose_latest) - int(video_seq)
        self.live_lag_sum += live_lag
        self.pose_lead_sum += pose_lead
        if overlay:
            self.overlay_ok += 1
        else:
            self.overlay_drop += 1
        if held:
            self.hold_show += 1
        seq_lag = int(video_seq) - int(pose_seq) if pose_seq >= 0 else -1
        if ui_idx % self.log_every == 0:
            avg_live = self.live_lag_sum / max(1, self.renders)
            avg_pose_lead = self.pose_lead_sum / max(1, self.renders)
            print(
                f"ui_sync render ui={ui_idx} path={path} video={video_seq} "
                f"pose={pose_seq} seq_lag={seq_lag} live={live_seq} "
                f"live−video={live_lag} pose_latest={pose_latest} "
                f"pose_latest−video={pose_lead} overlay={int(overlay)} "
                f"held={int(held)} | avg_live−video={avg_live:.1f} "
                f"avg_pose_lead={avg_pose_lead:.1f} "
                f"overlay_ok={self.overlay_ok} drop={self.overlay_drop} "
                f"skip_video_only={self.skipped_video_only}",
                flush=True,
            )
        if seq_lag != 0 and pose_seq >= 0:
            print(
                f"ui_sync WARN seq_mismatch ui={ui_idx} video={video_seq} "
                f"pose={pose_seq} seq_lag={seq_lag} overlay={int(overlay)}",
                flush=True,
            )

    def note_hold(
        self,
        *,
        ui_idx: int,
        hold_video_seq: int,
        hold_pose_seq: int,
        live_seq: int,
        pose_latest: int,
    ) -> None:
        if not ui_sync_diag_enabled():
            return
        self.hold_show += 1
        if ui_idx % self.log_every == 0:
            print(
                f"ui_sync hold ui={ui_idx} hold_video={hold_video_seq} "
                f"hold_pose={hold_pose_seq} live={live_seq} "
                f"live−hold={int(live_seq) - int(hold_video_seq)} "
                f"pose_latest={pose_latest}",
                flush=True,
            )
