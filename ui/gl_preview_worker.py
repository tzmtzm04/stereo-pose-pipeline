"""UI 预览层（P4）：EGL+GL NV12 fd 直显 + GL 2D 骨架，替代 imshow。"""

from __future__ import annotations

import threading
import time
from typing import TextIO

from app.route2.session_config import Route2SessionConfig
from core.display.display_tick import DisplayFrameSlot, DisplayTick
from core.display.gl_pose_overlay import apply_gl_pose_overlay
from core.display.gpu_display_preview import preview_eye_shape
from core.display.live_compose import (
    DisplayPoseResolver,
    ResolvedDisplayPose,
)
from core.display.pose_store import PoseStore
from core.display.ui_display_lag import UiLagBuffer
from core.display.video_ring import DisplayVideoRing, release_display_tick
from domain.l3.runtime import L3Runtime
from core.display.display_eye import gl_sbs_eye_mode
from orchestration.display_ui import build_status_line
from orchestration.metrics import (
    ThroughputTracker,
    apply_pose_infer_timing,
    write_frame_metrics,
    z_mean_cm,
)
from orchestration.types import FrameTiming
from platform.jetson.nv12_gl_display import Nv12GlDisplay
from ui.pose_driven_ui import (
    apply_pose_driven_advance,
    relieve_nv12_ring_pressure,
    step_pose_driven_ui,
    wait_pose_driven_bootstrap,
)
from ui.pose_overlay_hold import PoseOverlayHoldState, should_freeze_pose_overlay
from ui.render_sync_diag import RenderSyncDiag, ui_sync_diag_enabled
from ui.ui_lag_frame import lag_buffer_status, step_lagged_ui_frame

UI_SHOW_STALL_WARN_MS = 80.0
UI_WAIT_STATUS_EVERY = 40
_VIDEO_MISS_LOG_EVERY = 60

_video_miss_log = 0


def run_ui_gl_preview_worker(
    cfg: Route2SessionConfig,
    frame_slot: DisplayFrameSlot,
    store: PoseStore,
    stop: threading.Event,
    quit_evt: threading.Event,
    metrics_fp: TextIO | None,
    *,
    l3: L3Runtime | None = None,
    producer_th: threading.Thread | None = None,
    gl: Nv12GlDisplay | None = None,
    video_ring: DisplayVideoRing | None = None,
) -> None:
    ui_throughput = ThroughputTracker()
    ui_idx = 0
    prev_ui_end = time.perf_counter()
    wait_empty = 0
    use_lag = int(cfg.display_ui_lag_frames) > 0
    use_pose_driven = video_ring is not None and not use_lag

    wait_ms = int(cfg.display_pose_ui_wait_ms * 1000)
    if use_lag:
        boot = f"ui EGL+GL 预览启动 ui_lag={cfg.display_ui_lag_frames}f"
    elif use_pose_driven:
        boot = (
            f"ui EGL+GL 预览启动 pose→video "
            f"video_wait={wait_ms}ms（同 seq 叠图；首检前可原画，"
            f"首检后空窗 freeze 不追 live）"
        )
    else:
        boot = (
            f"ui EGL+GL 预览启动 ui_wait={wait_ms}ms"
            f"（无检测显示原画，有检测 exact 对齐）"
        )
    print(boot, flush=True)

    owns_gl = gl is None
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
    last_version = 0
    if gl is None:
        from core.display.hw_gl_display import create_hw_gl_display

        gl = create_hw_gl_display(cfg)
    if use_pose_driven and video_ring is not None:
        last_shown_seq = wait_pose_driven_bootstrap(store, video_ring)
        print(
            f"ui pose→video bootstrap last_shown={last_shown_seq} "
            f"live={video_ring.latest_capture_seq} "
            f"pose_latest={store.latest_capture_seq}",
            flush=True,
        )
    _embed_log_seq = -1
    try:
        while not quit_evt.is_set() and not stop.is_set():
            # 嵌入预览：Host 只排队几何，必须在本线程（持 Display）执行 X11
            if cfg.embedded_preview and gl is not None:
                if gl.apply_pending_embed():
                    gx, gy, gw, gh = gl.window_geometry()
                    if gl.applied_embed_seq != _embed_log_seq:
                        _embed_log_seq = gl.applied_embed_seq
                        mode = gl.embed_mode
                        print(
                            f"embed ui mode={mode} geom={gw}x{gh}+{gx}+{gy} "
                            f"renders_since_geom={gl.renders_since_geom}",
                            flush=True,
                        )

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
                    and overlay_hold.has_gl_hold()
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
                    resolved = ResolvedDisplayPose(
                        pose=overlay_hold.pose,
                        held=True,
                    )
                    quit = _render_gl_tick(
                        cfg,
                        gl,
                        overlay_hold.gl_tick,
                        resolved,
                        ui_idx,
                        prev_ui_end,
                        ui_throughput,
                        metrics_fp,
                        slot_wait_ms=slot_wait_ms,
                        l3=l3,
                        live_capture_seq=step.live_capture_seq,
                        show_live_delta=True,
                        overlay_hold=True,
                        sync_diag=sync_diag,
                        pose_latest_seq=int(store.latest_capture_seq),
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
                    if gl.poll_quit():
                        quit_evt.set()
                        stop.set()
                        break
                    continue

                if step.state == "wait_pose" and not overlay_hold.armed:
                    # 首检前：尚无原画可显
                    wait_empty += 1
                    if video_ring.nv12_fd and wait_empty % 8 == 0:
                        pruned = relieve_nv12_ring_pressure(video_ring, keep=2)
                        if pruned > 0:
                            print(
                                f"ui nv12 减压 prune={pruned} "
                                f"ring={video_ring.len()} live={step.live_capture_seq}",
                                flush=True,
                            )
                    if wait_empty == 1 or wait_empty % UI_WAIT_STATUS_EVERY == 0:
                        prod_ok, pull_fail = _prod_stats(frame_slot, video_ring)
                        print(
                            f"ui wait pose empty={wait_empty} prod_ok={prod_ok} "
                            f"pull_fail={pull_fail} live={step.live_capture_seq}",
                            flush=True,
                        )
                    if last_shown_seq < 0 and wait_empty % UI_WAIT_STATUS_EVERY == 0:
                        gl.set_status("等待 infer…  q quit")
                    if gl.poll_quit():
                        quit_evt.set()
                        stop.set()
                        break
                    continue

                if step.state == "wait_video":
                    _log_video_miss(step.target_seq, step.live_capture_seq)
                    if gl.poll_quit():
                        quit_evt.set()
                        stop.set()
                        break
                    continue

                if step.state == "skip_stale":
                    last_shown_seq = apply_pose_driven_advance(
                        last_shown_seq, step
                    )
                    if gl.poll_quit():
                        quit_evt.set()
                        stop.set()
                        break
                    continue

                if step.state == "wait_pose":
                    if gl.poll_quit():
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
                    if gl.poll_quit():
                        quit_evt.set()
                        stop.set()
                        break
                    continue

                last_shown_seq = apply_pose_driven_advance(last_shown_seq, step)
                render_tick = step.tick
                tick_owned = True
                try:
                    resolved = step.resolved or ResolvedDisplayPose(pose=None)
                    if (
                        step.state == "render"
                        and resolved.pose is not None
                        and render_tick is not None
                    ):
                        overlay_hold.save_gl_skeleton(
                            render_tick,
                            pose=resolved.pose,
                        )
                        tick_owned = False
                    quit = _render_gl_tick(
                        cfg,
                        gl,
                        render_tick,
                        resolved,
                        ui_idx,
                        prev_ui_end,
                        ui_throughput,
                        metrics_fp,
                        slot_wait_ms=slot_wait_ms,
                        l3=l3,
                        live_capture_seq=step.live_capture_seq,
                        show_live_delta=True,
                        sync_diag=sync_diag,
                        pose_latest_seq=int(store.latest_capture_seq),
                    )
                finally:
                    if tick_owned and render_tick is not None:
                        release_display_tick(render_tick)
            elif use_lag and lag_buf is not None:
                from core.display.display_tick import acquire_latest_for_ui

                version, tick, slot_wait_ms = acquire_latest_for_ui(
                    frame_slot,
                    last_version,
                    wait_timeout=0.008,
                )
                if tick is None:
                    if gl.poll_quit():
                        quit_evt.set()
                        stop.set()
                        break
                    continue
                last_version = version
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
                    if msg:
                        gl.set_status(msg)
                    if gl.poll_quit():
                        quit_evt.set()
                        stop.set()
                        break
                    continue
                quit = _render_gl_tick(
                    cfg,
                    gl,
                    display_tick,
                    resolved,
                    ui_idx,
                    prev_ui_end,
                    ui_throughput,
                    metrics_fp,
                    slot_wait_ms=slot_wait_ms,
                    l3=l3,
                    live_capture_seq=live_seq,
                    show_live_delta=True,
                )
            else:
                from core.display.display_tick import acquire_latest_for_ui

                version, tick, slot_wait_ms = acquire_latest_for_ui(
                    frame_slot,
                    last_version,
                    wait_timeout=0.008,
                )
                if tick is None:
                    if gl.poll_quit():
                        quit_evt.set()
                        stop.set()
                        break
                    continue
                last_version = version
                resolved = pose_resolver.resolve(store, int(tick.capture_seq))
                quit = _render_gl_tick(
                    cfg,
                    gl,
                    tick,
                    resolved,
                    ui_idx,
                    prev_ui_end,
                    ui_throughput,
                    metrics_fp,
                    slot_wait_ms=slot_wait_ms,
                    l3=l3,
                    live_capture_seq=int(tick.capture_seq),
                    show_live_delta=False,
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
        if owns_gl:
            gl.close()


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


def _render_gl_tick(
    cfg: Route2SessionConfig,
    gl: Nv12GlDisplay,
    tick: DisplayTick,
    resolved: ResolvedDisplayPose,
    ui_idx: int,
    prev_ui_end: float,
    ui_throughput: ThroughputTracker,
    metrics_fp: TextIO | None,
    *,
    slot_wait_ms: float,
    l3: L3Runtime | None,
    live_capture_seq: int,
    show_live_delta: bool,
    overlay_hold: bool = False,
    sync_diag: RenderSyncDiag | None = None,
    pose_latest_seq: int = -1,
) -> bool:
    frame = tick.nv12
    if frame is None or frame.dmabuf_fd is None:
        return False

    capture_seq = int(tick.capture_seq)
    pose = resolved.pose
    pose_held = resolved.held

    video_h, video_w = preview_eye_shape(frame)
    gl.set_sbs_eye(gl_sbs_eye_mode(cfg, is_sbs=bool(tick.nv12_sbs)))

    t_draw = time.perf_counter()
    if pose is not None and int(pose.capture_seq) == capture_seq:
        apply_gl_pose_overlay(gl, pose, cfg, video_w=video_w, video_h=video_h)
    else:
        gl.clear_pose()
    draw_ms = (time.perf_counter() - t_draw) * 1000.0

    infer_seq = pose.infer_seq if pose is not None else -1
    pose_seq = pose.capture_seq if pose is not None else -1
    seq_lag = capture_seq - int(pose_seq) if pose is not None else -1
    tri_ok = pose.tri_ok if pose is not None else False
    # 嵌入预览：每帧 XDrawString 状态栏也是 X11，关掉以稳住 ui_fps
    if cfg.embedded_preview:
        if ui_idx == 0:
            gl.set_status("")
    else:
        status = build_status_line(
            cfg,
            l3,
            frame_idx=ui_idx,
            capture_seq=capture_seq,
            infer_seq=infer_seq,
            tri_ok=tri_ok,
        )
        gl.set_status(status)

    t_show = time.perf_counter()
    show_ms = float(gl.render_fd(int(frame.dmabuf_fd)))
    _ = time.perf_counter() - t_show
    if cfg.embedded_preview:
        gl.note_render_ok()
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
        metrics_stage="ui_gl",
        ui_fps=ui_fps,
    )
    if ui_idx % cfg.log_every == 0:
        win_fps = ui_throughput.interval_fps(ui_idx + 1)
        lag_tag = ""
        if show_live_delta:
            lag_tag = f" live={live_capture_seq} lag={int(live_capture_seq) - capture_seq}"
        print(
            f"ui={ui_idx} ui_fps={ui_fps:.1f} win_fps={win_fps:.1f} "
            f"Z_mean={z:.1f}cm seq={capture_seq}{lag_tag} "
            f"pose_seq={pose_seq} seq_lag={seq_lag} "
            f"match_infer={infer_seq} hold={int(pose_held)} "
            f"overlay_hold={int(overlay_hold)} "
            f"wait={slot_wait_ms:.1f} draw={draw_ms:.1f} gl={show_ms:.1f} "
            f"ui_ms={ui_ms:.1f}",
            flush=True,
        )
    elif show_ms >= UI_SHOW_STALL_WARN_MS:
        print(
            f"ui stall warn ui={ui_idx} gl={show_ms:.1f}ms seq={capture_seq}",
            flush=True,
        )
    overlay = pose is not None and int(pose.capture_seq) == capture_seq
    if sync_diag is not None:
        sync_diag.note_render(
            ui_idx=ui_idx,
            path="gl",
            video_seq=capture_seq,
            pose_seq=int(pose_seq),
            live_seq=int(live_capture_seq),
            pose_latest=int(pose_latest_seq),
            overlay=overlay,
            held=bool(pose_held),
        )

    return bool(gl.poll_quit())
