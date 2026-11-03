"""UI 预览层：骨架驱动 VideoRing 或 legacy latest slot → overlay → imshow。"""

from __future__ import annotations

import threading
import time
from typing import TextIO

import numpy as np

from app.route2.session_config import Route2SessionConfig
from core.display.display_tick import (
    DisplayFrameSlot,
    DisplayTick,
    acquire_latest_for_ui,
)
from core.display.live_compose import (
    DisplayPoseResolver,
    OverlayFn,
    ResolvedDisplayPose,
    compose_display_frame,
)
from core.display.pose_store import PoseStore
from core.display.ui_display_lag import UiLagBuffer
from core.display.video_ring import DisplayVideoRing, release_display_tick
from domain.l3.runtime import L3Runtime
from orchestration.display_ui import (
    apply_pending_display_embed,
    build_status_line,
    embedded_canvas_size,
    poll_quit_key,
    show_frame,
    show_waiting_status,
    teardown_display_window,
)
from orchestration.metrics import (
    ThroughputTracker,
    apply_pose_infer_timing,
    write_frame_metrics,
    z_mean_cm,
)
from orchestration.types import FrameTiming
from ui.pose_driven_ui import (
    apply_pose_driven_advance,
    relieve_nv12_ring_pressure,
    step_pose_driven_ui,
    wait_pose_driven_bootstrap,
)
from ui.pose_overlay_hold import PoseOverlayHoldState, should_freeze_pose_overlay
from ui.render_sync_diag import RenderSyncDiag, ui_sync_diag_enabled
from ui.overlay_dump import get_overlay_dump_session
from ui.ui_lag_frame import lag_buffer_status, step_lagged_ui_frame

FRAME_SLOT_WAIT_SEC = 0.008
UI_SHOW_STALL_WARN_MS = 80.0
UI_WAIT_STATUS_EVERY = 40
_VIDEO_MISS_LOG_EVERY = 60

_video_miss_log = 0


def run_ui_preview_worker(
    cfg: Route2SessionConfig,
    frame_slot: DisplayFrameSlot,
    store: PoseStore,
    stop: threading.Event,
    quit_evt: threading.Event,
    metrics_fp: TextIO | None,
    *,
    l3: L3Runtime | None = None,
    overlay_fn: OverlayFn | None = None,
    producer_th: threading.Thread | None = None,
    video_ring: DisplayVideoRing | None = None,
) -> None:
    ui_throughput = ThroughputTracker()
    ui_idx = 0
    prev_ui_end = time.perf_counter()
    last_version = 0
    wait_empty = 0
    use_lag = int(cfg.display_ui_lag_frames) > 0
    use_pose_driven = video_ring is not None and not use_lag

    wait_ms = int(cfg.display_pose_ui_wait_ms * 1000)
    if use_lag:
        boot = f"ui preview 线程启动 ui_lag={cfg.display_ui_lag_frames}f"
    elif use_pose_driven:
        boot = (
            f"ui preview 线程启动 pose→video "
            f"video_wait={wait_ms}ms（同 seq 叠图；首检前可原画，"
            f"首检后空窗 freeze 不追 live）"
        )
    else:
        boot = (
            f"ui preview 线程启动 ui_wait={wait_ms}ms"
            f"（无检测显示原画，有检测 exact 对齐）"
        )
    print(boot, flush=True)

    pose_resolver = DisplayPoseResolver(cfg)
    lag_buf = UiLagBuffer(cfg.display_ui_lag_frames) if use_lag else None
    overlay_hold = PoseOverlayHoldState()
    sync_diag = RenderSyncDiag.from_env()
    if ui_sync_diag_enabled():
        print(
            f"ui_sync 诊断开 every={sync_diag.log_every} "
            f"(ROUTE2_UI_SYNC_DIAG=0 关)",
            flush=True,
        )
    last_shown_seq = -1
    if use_pose_driven and video_ring is not None:
        last_shown_seq = wait_pose_driven_bootstrap(store, video_ring)
        print(
            f"ui pose→video bootstrap last_shown={last_shown_seq} "
            f"live={video_ring.latest_capture_seq} "
            f"pose_latest={store.latest_capture_seq}",
            flush=True,
        )
    try:
        while not quit_evt.is_set() and not stop.is_set():
            # fall_risk 嵌套：Host 排队的几何必须在本线程 Apply
            if cfg.embedded_preview:
                apply_pending_display_embed()

            if producer_th is not None and not producer_th.is_alive():
                prod_ok, pull_fail = _prod_stats(frame_slot, video_ring)
                print(
                    f"ui 退出: display producer 已停止 ok={prod_ok} "
                    f"pull_fail={pull_fail}",
                    flush=True,
                )
                stop.set()
                break

            if use_pose_driven and video_ring is not None:
                step = step_pose_driven_ui(
                    cfg,
                    store,
                    video_ring,
                    pose_resolver,
                    last_shown_seq,
                    hold_armed=overlay_hold.armed,
                )
                slot_wait_ms = step.pose_wait_ms + step.video_wait_ms
                sync_diag.note_step(
                    step,
                    ui_idx=ui_idx,
                    pose_latest=int(store.latest_capture_seq),
                )

                if (
                    should_freeze_pose_overlay(step, overlay_hold)
                    and overlay_hold.has_hold()
                ):
                    if step.tick is not None:
                        release_display_tick(step.tick)
                    sync_diag.note_hold(
                        ui_idx=ui_idx,
                        hold_video_seq=int(overlay_hold.capture_seq),
                        hold_pose_seq=int(
                            overlay_hold.pose.capture_seq
                            if overlay_hold.pose is not None
                            else -1
                        ),
                        live_seq=step.live_capture_seq,
                        pose_latest=int(store.latest_capture_seq),
                    )
                    quit = _show_frozen_ui_tick(
                        cfg,
                        overlay_hold,
                        ui_idx,
                        prev_ui_end,
                        ui_throughput,
                        metrics_fp,
                        slot_wait_ms=slot_wait_ms,
                        l3=l3,
                        live_capture_seq=step.live_capture_seq,
                    )
                    if step.state == "skip_stale":
                        last_shown_seq = apply_pose_driven_advance(
                            last_shown_seq, step
                        )
                    prev_ui_end = time.perf_counter()
                    ui_idx += 1
                    if quit:
                        quit_evt.set()
                        stop.set()
                        break
                    if poll_quit_key(cfg):
                        quit_evt.set()
                        stop.set()
                        break
                    continue

                if step.state == "wait_pose" and not overlay_hold.armed:
                    # 首检前：尚无原画可显
                    wait_empty += 1
                    if video_ring.nv12_fd and wait_empty % 8 == 0:
                        relieve_nv12_ring_pressure(video_ring, keep=2)
                    if wait_empty == 1 or wait_empty % UI_WAIT_STATUS_EVERY == 0:
                        prod_ok, pull_fail = _prod_stats(frame_slot, video_ring)
                        print(
                            f"ui wait pose empty={wait_empty} prod_ok={prod_ok} "
                            f"pull_fail={pull_fail} live={step.live_capture_seq}",
                            flush=True,
                        )
                    if last_shown_seq < 0 and wait_empty % UI_WAIT_STATUS_EVERY == 0:
                        msg = (
                            f"等待 infer…\n"
                            f"prod_ok={prod_ok} pull_fail={pull_fail}"
                        )
                        if show_waiting_status(cfg, msg):
                            quit_evt.set()
                            stop.set()
                            break
                    elif poll_quit_key(cfg):
                        quit_evt.set()
                        stop.set()
                        break
                    continue

                if step.state == "wait_video":
                    _log_video_miss(step.target_seq, step.live_capture_seq)
                    if poll_quit_key(cfg):
                        quit_evt.set()
                        stop.set()
                        break
                    continue

                if step.state == "skip_stale":
                    last_shown_seq = apply_pose_driven_advance(
                        last_shown_seq, step
                    )
                    if poll_quit_key(cfg):
                        quit_evt.set()
                        stop.set()
                        break
                    continue

                if step.state == "wait_pose":
                    if poll_quit_key(cfg):
                        quit_evt.set()
                        stop.set()
                        break
                    continue

                wait_empty = 0
                # pose→video：render 始终可显；render_video_only 仅首检前（防原画反压）
                can_show = step.tick is not None and (
                    step.state == "render"
                    or (
                        step.state == "render_video_only"
                        and not overlay_hold.armed
                    )
                )
                if not can_show:
                    if step.tick is not None:
                        release_display_tick(step.tick)
                    if poll_quit_key(cfg):
                        quit_evt.set()
                        stop.set()
                        break
                    continue

                last_shown_seq = apply_pose_driven_advance(last_shown_seq, step)
                render_tick = step.tick
                dump_sess = get_overlay_dump_session()
                if (
                    dump_sess is not None
                    and render_tick is not None
                    and render_tick.bgr is not None
                ):
                    pose_seq = -1
                    if step.resolved is not None and step.resolved.pose is not None:
                        pose_seq = int(step.resolved.pose.capture_seq)
                    dump_sess.dump_ring_take(
                        ui_idx=ui_idx,
                        video_seq=int(render_tick.capture_seq),
                        ring_bgr=np.ascontiguousarray(render_tick.bgr).copy(),
                        pose_capture_seq=pose_seq,
                    )
                try:
                    resolved = step.resolved or ResolvedDisplayPose(pose=None)
                    quit = _render_ui_tick(
                        cfg,
                        render_tick,
                        resolved,
                        ui_idx,
                        prev_ui_end,
                        ui_throughput,
                        metrics_fp,
                        slot_wait_ms=slot_wait_ms,
                        l3=l3,
                        overlay_fn=overlay_fn,
                        live_capture_seq=step.live_capture_seq,
                        show_live_delta=True,
                        overlay_hold=overlay_hold,
                        pose_latest_seq=int(store.latest_capture_seq),
                        sync_diag=sync_diag,
                    )
                finally:
                    release_display_tick(render_tick)
            else:
                version, tick, slot_wait_ms = acquire_latest_for_ui(
                    frame_slot,
                    last_version,
                    wait_timeout=FRAME_SLOT_WAIT_SEC,
                )

                if tick is None:
                    wait_empty += 1
                    if wait_empty == 1 or wait_empty % UI_WAIT_STATUS_EVERY == 0:
                        prod_ok, pull_fail = _prod_stats(frame_slot, video_ring)
                        print(
                            f"ui wait empty={wait_empty} prod_ok={prod_ok} "
                            f"pull_fail={pull_fail}",
                            flush=True,
                        )
                    if wait_empty % UI_WAIT_STATUS_EVERY == 0:
                        prod_ok, pull_fail = _prod_stats(frame_slot, video_ring)
                        msg = (
                            f"等待 display 生产线…\n"
                            f"prod_ok={prod_ok} pull_fail={pull_fail}\n"
                            f"infer 在跑但 UI 无帧时请查 prod pull_fail"
                        )
                        if show_waiting_status(cfg, msg):
                            quit_evt.set()
                            stop.set()
                            break
                    elif poll_quit_key(cfg):
                        quit_evt.set()
                        stop.set()
                        break
                    continue

                wait_empty = 0
                last_version = version

                if use_lag and lag_buf is not None:
                    live_seq = int(tick.capture_seq)
                    state, display_tick, resolved, live_seq = step_lagged_ui_frame(
                        cfg,
                        store,
                        lag_buf,
                        pose_resolver,
                        tick,
                        live_capture_seq=live_seq,
                    )
                    if state in ("filling", "skip") or display_tick is None or resolved is None:
                        msg = lag_buffer_status(cfg, lag_buf)
                        if msg and show_waiting_status(cfg, msg):
                            quit_evt.set()
                            stop.set()
                            break
                        if poll_quit_key(cfg):
                            quit_evt.set()
                            stop.set()
                            break
                        continue
                    render_tick = display_tick
                    live_seq_arg = live_seq
                    show_live_delta = True
                else:
                    render_tick = tick
                    resolved = pose_resolver.resolve(store, int(tick.capture_seq))
                    live_seq_arg = int(tick.capture_seq)
                    show_live_delta = False

                quit = _render_ui_tick(
                    cfg,
                    render_tick,
                    resolved,
                    ui_idx,
                    prev_ui_end,
                    ui_throughput,
                    metrics_fp,
                    slot_wait_ms=slot_wait_ms,
                    l3=l3,
                    overlay_fn=overlay_fn,
                    live_capture_seq=live_seq_arg,
                    show_live_delta=show_live_delta,
                )

            prev_ui_end = time.perf_counter()
            ui_idx += 1
            if quit:
                quit_evt.set()
                stop.set()
                break
            if quit_evt.is_set() or stop.is_set():
                break
    finally:
        overlay_hold.release_gl_tick()
        teardown_display_window()


def _prod_stats(
    frame_slot: DisplayFrameSlot,
    video_ring: DisplayVideoRing | None,
) -> tuple[int, int]:
    if video_ring is not None:
        return video_ring.prod_stats()
    return frame_slot.prod_stats()


def _log_video_miss(target_seq: int, live_seq: int) -> None:
    global _video_miss_log
    _video_miss_log += 1
    if _video_miss_log == 1 or _video_miss_log % _VIDEO_MISS_LOG_EVERY == 0:
        print(
            f"ui pose→video 原画未就绪 seq={target_seq} live={live_seq} "
            f"(累计={_video_miss_log})",
            flush=True,
        )


def _show_frozen_ui_tick(
    cfg: Route2SessionConfig,
    hold: PoseOverlayHoldState,
    ui_idx: int,
    prev_ui_end: float,
    ui_throughput: ThroughputTracker,
    metrics_fp: TextIO | None,
    *,
    slot_wait_ms: float,
    l3: L3Runtime | None,
    live_capture_seq: int,
) -> bool:
    capture_seq = int(hold.capture_seq)
    pose = hold.pose
    t_show = time.perf_counter()
    quit = show_frame(cfg, hold.composite, hold.status)
    show_ms = (time.perf_counter() - t_show) * 1000.0
    ui_ms = (time.perf_counter() - prev_ui_end) * 1000.0
    ui_throughput.mark()
    ui_fps = ui_throughput.rolling_fps()

    infer_seq = pose.infer_seq if pose is not None else -1
    pose_seq = pose.capture_seq if pose is not None else -1
    tri_ok = pose.tri_ok if pose is not None else False
    timing = FrameTiming()
    timing.display_ms = show_ms
    timing.queue_wait_ms = slot_wait_ms
    apply_pose_infer_timing(timing, pose)
    timing.e2e_ms = ui_ms
    z = z_mean_cm(pose.kpts3d if pose is not None else None, cfg.conf)
    write_frame_metrics(
        metrics_fp,
        ui_idx,
        timing,
        e2e_fps=ui_fps,
        tri_ok=tri_ok,
        z_mean=round(z, 1),
        capture_seq=capture_seq,
        metrics_stage="ui",
        ui_fps=ui_fps,
    )
    if ui_idx % cfg.log_every == 0:
        win_fps = ui_throughput.interval_fps(ui_idx + 1)
        lag_tag = f" live={live_capture_seq} lag={int(live_capture_seq) - capture_seq}"
        print(
            f"ui={ui_idx} ui_fps={ui_fps:.1f} win_fps={win_fps:.1f} "
            f"Z_mean={z:.1f}cm seq={capture_seq}{lag_tag} "
            f"pose_seq={pose_seq} seq_lag=0 "
            f"match_infer={infer_seq} hold=0 overlay_hold=1 "
            f"wait={slot_wait_ms:.1f} draw=0.0 show={show_ms:.1f} "
            f"ui_ms={ui_ms:.1f}",
            flush=True,
        )
    elif show_ms >= UI_SHOW_STALL_WARN_MS:
        print(
            f"ui stall warn ui={ui_idx} show={show_ms:.1f}ms seq={capture_seq} "
            f"overlay_hold=1",
            flush=True,
        )
    return bool(quit)


def _render_ui_tick(
    cfg: Route2SessionConfig,
    tick: DisplayTick,
    resolved: ResolvedDisplayPose,
    ui_idx: int,
    prev_ui_end: float,
    ui_throughput: ThroughputTracker,
    metrics_fp: TextIO | None,
    *,
    slot_wait_ms: float,
    l3: L3Runtime | None,
    overlay_fn: OverlayFn | None,
    live_capture_seq: int,
    show_live_delta: bool,
    overlay_hold: PoseOverlayHoldState | None = None,
    pose_latest_seq: int = -1,
    sync_diag: RenderSyncDiag | None = None,
) -> bool:
    capture_seq = int(tick.capture_seq)
    pose = resolved.pose
    pose_held = resolved.held

    if tick.bgr is None:
        return False

    video_bgr = np.ascontiguousarray(tick.bgr).copy()

    t_draw = time.perf_counter()
    overlay_pose = pose
    if overlay_pose is None or int(overlay_pose.capture_seq) != capture_seq:
        overlay_pose = None
    hole = embedded_canvas_size() if cfg.embedded_preview else None
    composite = compose_display_frame(
        cfg,
        video_bgr,
        overlay_pose,
        overlay_fn=overlay_fn,
        out_size=hole,
        fill="cover" if hole is not None else "fit",
    )
    draw_ms = (time.perf_counter() - t_draw) * 1000.0

    dump_sess = get_overlay_dump_session()
    if dump_sess is not None:
        dump_sess.dump_render(
            ui_idx=ui_idx,
            video_seq=capture_seq,
            cfg=cfg,
            video_bgr=video_bgr,
            composite_window=composite,
            pose=pose,
            overlay_pose=overlay_pose,
            overlay_fn=overlay_fn,
            live_capture_seq=live_capture_seq,
            pose_held=bool(pose_held),
            hole=hole,
        )

    infer_seq = pose.infer_seq if pose is not None else -1
    pose_seq = pose.capture_seq if pose is not None else -1
    seq_lag = capture_seq - int(pose_seq) if pose is not None else -1
    tri_ok = pose.tri_ok if pose is not None else False
    status = build_status_line(
        cfg,
        l3,
        frame_idx=ui_idx,
        capture_seq=capture_seq,
        infer_seq=infer_seq,
        tri_ok=tri_ok,
    )
    if overlay_pose is not None and overlay_hold is not None:
        overlay_hold.save_skeleton_frame(
            composite,
            status=status,
            pose=overlay_pose,
            capture_seq=capture_seq,
        )

    t_show = time.perf_counter()
    quit = show_frame(cfg, composite, status)
    show_ms = (time.perf_counter() - t_show) * 1000.0
    ui_ms = (time.perf_counter() - prev_ui_end) * 1000.0
    ui_throughput.mark()
    ui_fps = ui_throughput.rolling_fps()

    timing = FrameTiming()
    timing.draw_ms = draw_ms
    timing.display_ms = show_ms
    timing.queue_wait_ms = slot_wait_ms
    apply_pose_infer_timing(timing, pose)
    timing.e2e_ms = ui_ms
    z = z_mean_cm(pose.kpts3d if pose is not None else None, cfg.conf)
    write_frame_metrics(
        metrics_fp,
        ui_idx,
        timing,
        e2e_fps=ui_fps,
        tri_ok=tri_ok,
        z_mean=round(z, 1),
        capture_seq=capture_seq,
        metrics_stage="ui",
        ui_fps=ui_fps,
    )
    if ui_idx % cfg.log_every == 0:
        win_fps = ui_throughput.interval_fps(ui_idx + 1)
        lag_tag = ""
        if show_live_delta:
            lag_tag = f" live={live_capture_seq} lag={int(live_capture_seq) - capture_seq}"
            if pose_latest_seq >= 0:
                lag_tag += f" pose_latest={pose_latest_seq}"
        print(
            f"ui={ui_idx} ui_fps={ui_fps:.1f} win_fps={win_fps:.1f} "
            f"Z_mean={z:.1f}cm seq={capture_seq}{lag_tag} "
            f"pose_seq={pose_seq} seq_lag={seq_lag} "
            f"match_infer={infer_seq} hold={int(pose_held)} "
            f"wait={slot_wait_ms:.1f} draw={draw_ms:.1f} show={show_ms:.1f} "
            f"ui_ms={ui_ms:.1f}",
            flush=True,
        )
    elif show_ms >= UI_SHOW_STALL_WARN_MS:
        print(
            f"ui stall warn ui={ui_idx} show={show_ms:.1f}ms seq={capture_seq}",
            flush=True,
        )
    if sync_diag is not None:
        sync_diag.note_render(
            ui_idx=ui_idx,
            path="opencv",
            video_seq=capture_seq,
            pose_seq=int(pose_seq),
            live_seq=int(live_capture_seq),
            pose_latest=int(pose_latest_seq),
            overlay=overlay_pose is not None,
            held=bool(pose_held),
        )
    return bool(quit)
