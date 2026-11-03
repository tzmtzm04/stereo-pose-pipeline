"""UI 骨架驱动原画：最新可用骨架 + 同 capture_seq 原画 exact 叠图。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Literal

from app.route2.session_config import Route2SessionConfig
from core.display.display_tick import DisplayTick
from core.display.live_compose import (
    DisplayPoseResolver,
    ResolvedDisplayPose,
    resolve_pose_sample_for_display,
)
from core.display.pose_store import PoseDisplaySample, PoseStore
from core.display.video_ring import DisplayVideoRing, release_display_tick

POSE_POLL_SEC = 0.004
UI_POSE_CATCH_UP_FRAMES = 6
_STALE_SKIP_LOG_EVERY = 30

_stale_skip_log = 0


def ring_oldest_seq(live_capture_seq: int, ring_max_frames: int) -> int:
    """环内仍保留的最小 capture_seq。"""
    if live_capture_seq < 0:
        return -1
    return max(0, int(live_capture_seq) - int(ring_max_frames) + 1)


def bootstrap_last_shown_seq(store: PoseStore, video_ring: DisplayVideoRing) -> int:
    """启动对齐：从环内可显示的最新骨架前一位开始，避免从 seq=0 追 backlog。"""
    live = int(video_ring.latest_capture_seq)
    latest_pose = int(store.latest_capture_seq)
    if live < 0 and latest_pose < 0:
        return -1
    oldest = ring_oldest_seq(live, video_ring.max_frames) if live >= 0 else -1
    if latest_pose < 0:
        return max(-1, oldest - 1)
    if oldest >= 0 and latest_pose < oldest:
        return max(-1, oldest - 1)
    return max(-1, latest_pose - 1)


def _find_usable_pair(
    cfg: Route2SessionConfig,
    store: PoseStore,
    video_ring: DisplayVideoRing,
    after_seq: int,
) -> tuple[PoseDisplaySample | None, DisplayTick | None, int, int]:
    """同 seq 原画+骨架：默认顺序追帧；落后过多时 catch-up 到最新。

    快速运动时顺序播放避免跳帧卡顿；仅当落后超过 UI_POSE_CATCH_UP_FRAMES
    才回落到「最新可用对」以免 live 拉开太远。
    """
    live = int(video_ring.latest_capture_seq)
    latest_pose = int(store.latest_capture_seq)
    after = int(after_seq)

    if latest_pose < 0 and live < 0:
        return None, None, after, 0

    oldest = ring_oldest_seq(live, video_ring.max_frames) if live >= 0 else -1
    lo = max(oldest, after + 1) if oldest >= 0 else after + 1
    if lo < 0:
        lo = 0

    def _usable(pose: PoseDisplaySample | None) -> bool:
        if pose is None:
            return False
        return resolve_pose_sample_for_display(cfg, pose).pose is not None

    hi = latest_pose
    if hi < lo:
        return None, None, after, 0

    lag = max(0, hi - after) if after >= 0 else 0
    catch_up = lag > int(UI_POSE_CATCH_UP_FRAMES)
    seq_iter = (
        range(hi, lo - 1, -1)
        if catch_up
        else range(lo, hi + 1)
    )

    for seq in seq_iter:
        pose = store.match(seq, policy="exact")
        if not _usable(pose):
            continue
        tick = video_ring.get(seq)
        skipped = max(0, seq - after - 1) if after >= 0 else 0
        return pose, tick, seq, skipped

    return None, None, after, 0


def _find_latest_usable_pair(
    cfg: Route2SessionConfig,
    store: PoseStore,
    video_ring: DisplayVideoRing,
    after_seq: int,
) -> tuple[PoseDisplaySample | None, DisplayTick | None, int, int]:
    """兼容旧名；见 `_find_usable_pair`。"""
    return _find_usable_pair(cfg, store, video_ring, after_seq)


def _find_pose_with_video(
    cfg: Route2SessionConfig,
    store: PoseStore,
    video_ring: DisplayVideoRing,
    after_seq: int,
) -> tuple[PoseDisplaySample | None, DisplayTick | None, int, int]:
    """兼容旧名；见 `_find_usable_pair`。"""
    return _find_usable_pair(cfg, store, video_ring, after_seq)


def _log_stale_skip(
    *,
    from_seq: int,
    to_seq: int,
    live_seq: int,
    skipped: int,
) -> None:
    global _stale_skip_log
    if skipped <= 0:
        return
    _stale_skip_log += 1
    if _stale_skip_log == 1 or _stale_skip_log % _STALE_SKIP_LOG_EVERY == 0:
        print(
            f"ui pose→video 骨架过旧 seq={from_seq}..{to_seq} "
            f"live={live_seq} 跳过={skipped} (累计={_stale_skip_log})",
            flush=True,
        )


StepState = Literal[
    "render",
    "render_video_only",
    "hold_overlay",
    "wait_pose",
    "wait_video",
    "skip_stale",
]


@dataclass(frozen=True)
class PoseDrivenStep:
    state: StepState
    tick: DisplayTick | None = None
    resolved: ResolvedDisplayPose | None = None
    pose_wait_ms: float = 0.0
    video_wait_ms: float = 0.0
    target_seq: int = -1
    live_capture_seq: int = -1
    advance_last_shown_to: int = -1
    stale_skipped: int = 0


def wait_pose_driven_bootstrap(
    store: PoseStore,
    video_ring: DisplayVideoRing,
    *,
    timeout_sec: float = 2.0,
) -> int:
    """短等 prod/infer 就绪后计算 last_shown。"""
    deadline = time.monotonic() + max(0.1, float(timeout_sec))
    while time.monotonic() < deadline:
        if video_ring.latest_capture_seq >= 0 or store.latest_capture_seq >= 0:
            break
        time.sleep(0.005)
    last = bootstrap_last_shown_seq(store, video_ring)
    if video_ring.nv12_fd:
        video_ring.prune_keep_latest(2)
    return last


def _video_only_step(
    video_ring: DisplayVideoRing,
    last_shown_seq: int,
    *,
    live_seq: int,
    pose_latest_seq: int = -1,
    pose_wait_ms: float = 0.0,
    video_wait_ms: float = 0.0,
    stale_skipped: int = 0,
) -> PoseDrivenStep:
    """首检前：取原画直显；last_shown 不得甩开已有 pose（防反压环）。"""
    tick = try_video_only_fallback(video_ring, last_shown_seq)
    if tick is None:
        return PoseDrivenStep(
            state="wait_pose",
            pose_wait_ms=pose_wait_ms,
            video_wait_ms=video_wait_ms,
            live_capture_seq=live_seq,
            stale_skipped=stale_skipped,
        )
    seq = int(tick.capture_seq)
    # 有 pose 时 last_shown 不得超过 pose，否则 _find_usable_pair 永远 hi<lo
    if int(pose_latest_seq) >= 0:
        advance_to = min(seq, int(pose_latest_seq))
    else:
        # 尚无任何 pose：不推进 last_shown，避免之后永远追不上
        advance_to = int(last_shown_seq)
    return PoseDrivenStep(
        state="render_video_only",
        tick=tick,
        resolved=ResolvedDisplayPose(pose=None),
        pose_wait_ms=pose_wait_ms,
        video_wait_ms=video_wait_ms,
        target_seq=seq,
        live_capture_seq=live_seq,
        advance_last_shown_to=advance_to,
        stale_skipped=stale_skipped,
    )


def _hold_overlay_step(
    *,
    live_seq: int,
    pose_wait_ms: float = 0.0,
    video_wait_ms: float = 0.0,
    target_seq: int = -1,
    stale_skipped: int = 0,
) -> PoseDrivenStep:
    """首检后无可用骨架：不取原画，由 UI 冻结上一帧叠图。"""
    return PoseDrivenStep(
        state="hold_overlay",
        pose_wait_ms=pose_wait_ms,
        video_wait_ms=video_wait_ms,
        target_seq=target_seq,
        live_capture_seq=live_seq,
        stale_skipped=stale_skipped,
    )


def _no_pose_step(
    cfg: Route2SessionConfig,
    video_ring: DisplayVideoRing,
    last_shown_seq: int,
    *,
    live_seq: int,
    pose_wait_ms: float,
    hold_armed: bool,
    pose_latest_seq: int = -1,
) -> PoseDrivenStep:
    """infer 尚无下一帧 pose：首检前可原画；首检后只 hold（不追 live 反压）。"""
    _ = cfg
    if hold_armed:
        return _hold_overlay_step(
            live_seq=live_seq,
            pose_wait_ms=pose_wait_ms,
        )
    return _video_only_step(
        video_ring,
        int(last_shown_seq),
        live_seq=live_seq,
        pose_latest_seq=int(pose_latest_seq),
        pose_wait_ms=pose_wait_ms,
    )


def step_pose_driven_ui(
    cfg: Route2SessionConfig,
    store: PoseStore,
    video_ring: DisplayVideoRing,
    pose_resolver: DisplayPoseResolver,
    last_shown_seq: int,
    *,
    hold_armed: bool = False,
) -> PoseDrivenStep:
    """首检前可原画；首检后 pose→video exact，空窗 freeze 不追 live。"""
    _ = pose_resolver
    live_seq = int(video_ring.latest_capture_seq)
    pose_latest_seq = int(store.latest_capture_seq)

    t_pose = time.perf_counter()
    pose, tick, target_seq, stale_skipped = _find_latest_usable_pair(
        cfg,
        store,
        video_ring,
        int(last_shown_seq),
    )
    if pose is None:
        pose = store.wait_for_next_pose_after(
            int(last_shown_seq),
            timeout=POSE_POLL_SEC,
        )
        if pose is None:
            pose_wait_ms = (time.perf_counter() - t_pose) * 1000.0
            return _no_pose_step(
                cfg,
                video_ring,
                int(last_shown_seq),
                live_seq=live_seq,
                pose_wait_ms=pose_wait_ms,
                hold_armed=hold_armed,
                pose_latest_seq=pose_latest_seq,
            )
        pose, tick, target_seq, stale_skipped = _find_usable_pair(
            cfg,
            store,
            video_ring,
            int(last_shown_seq),
        )
        if pose is None:
            pose_wait_ms = (time.perf_counter() - t_pose) * 1000.0
            return _no_pose_step(
                cfg,
                video_ring,
                int(last_shown_seq),
                live_seq=live_seq,
                pose_wait_ms=pose_wait_ms,
                hold_armed=hold_armed,
                pose_latest_seq=pose_latest_seq,
            )

    pose_wait_ms = (time.perf_counter() - t_pose) * 1000.0
    if stale_skipped > 0:
        _log_stale_skip(
            from_seq=int(last_shown_seq),
            to_seq=target_seq,
            live_seq=live_seq,
            skipped=stale_skipped,
        )

    if tick is None:
        tick, video_wait_ms = video_ring.wait_and_take(
            target_seq,
            timeout=float(cfg.display_pose_ui_wait_ms),
        )
        if tick is None:
            live_seq = int(video_ring.latest_capture_seq)
            oldest = ring_oldest_seq(live_seq, video_ring.max_frames)
            if oldest >= 0 and target_seq < oldest:
                return PoseDrivenStep(
                    state="skip_stale",
                    target_seq=target_seq,
                    live_capture_seq=live_seq,
                    advance_last_shown_to=target_seq,
                    stale_skipped=stale_skipped,
                    pose_wait_ms=pose_wait_ms,
                    video_wait_ms=video_wait_ms,
                )
            return PoseDrivenStep(
                state="wait_video",
                target_seq=target_seq,
                live_capture_seq=live_seq,
                pose_wait_ms=pose_wait_ms,
                video_wait_ms=video_wait_ms,
                stale_skipped=stale_skipped,
                resolved=resolve_pose_sample_for_display(cfg, pose),
            )
    else:
        video_wait_ms = 0.0
        taken = video_ring.take(target_seq)
        if taken is not None:
            tick = taken

    resolved = resolve_pose_sample_for_display(cfg, pose)
    if resolved.pose is None:
        if hold_armed:
            if tick is not None:
                release_display_tick(tick)
            return _hold_overlay_step(
                live_seq=live_seq,
                pose_wait_ms=pose_wait_ms,
                video_wait_ms=video_wait_ms,
                target_seq=target_seq,
                stale_skipped=stale_skipped + 1,
            )
        advance_to = (
            min(int(target_seq), pose_latest_seq)
            if pose_latest_seq >= 0
            else int(last_shown_seq)
        )
        return PoseDrivenStep(
            state="render_video_only",
            tick=tick,
            resolved=ResolvedDisplayPose(pose=None),
            pose_wait_ms=pose_wait_ms,
            video_wait_ms=video_wait_ms,
            target_seq=target_seq,
            live_capture_seq=live_seq,
            advance_last_shown_to=advance_to,
            stale_skipped=stale_skipped + 1,
        )

    return PoseDrivenStep(
        state="render",
        tick=tick,
        resolved=resolved,
        pose_wait_ms=pose_wait_ms,
        video_wait_ms=video_wait_ms,
        target_seq=target_seq,
        live_capture_seq=live_seq,
        advance_last_shown_to=target_seq,
        stale_skipped=stale_skipped,
    )


def try_video_only_fallback(
    video_ring: DisplayVideoRing,
    last_shown_seq: int,
) -> DisplayTick | None:
    """infer 无 pose 时取出最新原画（须用后 release）。"""
    return video_ring.take_newest_after(int(last_shown_seq))


def peek_video_only_fallback(
    video_ring: DisplayVideoRing,
    after_seq: int,
) -> DisplayTick | None:
    """infer 尚未就绪时只读最新原画（不 take，不 release）。"""
    return video_ring.peek_newest_after(int(after_seq))


def relieve_nv12_ring_pressure(video_ring: DisplayVideoRing, *, keep: int = 2) -> int:
    if not video_ring.nv12_fd:
        return 0
    return video_ring.prune_keep_latest(keep)


def apply_pose_driven_advance(last_shown_seq: int, step: PoseDrivenStep) -> int:
    if step.advance_last_shown_to >= 0:
        return int(step.advance_last_shown_to)
    return int(last_shown_seq)
