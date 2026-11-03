"""推理 consumer：prep 信号唤醒 → prep_q 最新帧 → TRT+post，或 capture_q 直推。"""

from __future__ import annotations

import queue
import threading
import traceback
from collections import deque
from typing import TYPE_CHECKING, Callable, TextIO

from app.route2.session_config import Route2SessionConfig
from core.display.pose_store import PoseStore
from core.infer.pose3d_pipeline import StereoPose3DPipeline
from domain.l1.runtime import L1Runtime
from domain.l3.runtime import L3Runtime
from orchestration.constants import ROLLING_N
from orchestration.metrics import (
    ThroughputTracker,
    hist_mean,
    write_frame_metrics,
    z_mean_cm,
)
from orchestration.packet_builder import build_display_packet
from orchestration.pose_publish import publish_infer_pose
from orchestration.queue_util import take_latest_from_queue, try_take_latest_from_queue
from orchestration.types import FrameTiming
from orchestration.vic_bgr_pipeline import vic_bgr_prep_pipeline_enabled

if TYPE_CHECKING:
    from orchestration.c2_auto_stop import C2AutoStopGate


def _publish_pose_and_metrics(
    *,
    cfg: Route2SessionConfig,
    pose,
    frame_idx: int,
    throughput: ThroughputTracker,
    e2e_hist: deque,
    par_egl_hist: deque,
    par_lb_hist: deque,
    par_trt_hist: deque,
    par_post_hist: deque,
    par_drain_hist: deque,
    par_wait_hist: deque,
    par_unmap_hist: deque,
    par_trt_sync_hist: deque,
    tri_hist: deque,
    vic_hist: deque,
    cap_skip_hist: deque,
    prep_skip_hist: deque,
    pose_store: PoseStore | None,
    l1: L1Runtime | None,
    l3: L3Runtime | None,
    metrics_fp: TextIO | None,
    on_frame: Callable[[], None] | None,
    c2_auto_stop: C2AutoStopGate | None = None,
) -> int:
    if on_frame is not None:
        on_frame()
    throughput.mark()

    split_display = pose_store is not None
    if split_display:
        publish_infer_pose(
            pose,
            pose_store,
            l1=l1,
            l3=l3,
            c2_auto_stop=c2_auto_stop,
        )
        e2e_ms = (
            pose.infer.capture_ms
            + pose.infer.infer_parallel_ms
            + pose.tri_ms
        )
        e2e_hist.append(e2e_ms)
        bd = pose.infer.par_breakdown()
        par_egl_hist.append(bd.egl_ms)
        par_lb_hist.append(bd.lb_ms)
        par_trt_hist.append(bd.trt_ms)
        par_post_hist.append(bd.post_ms)
        par_drain_hist.append(bd.drain_ms)
        par_wait_hist.append(bd.wait_ms)
        par_unmap_hist.append(bd.unmap_ms)
        par_trt_sync_hist.append(bd.trt_d2h_sync_ms)
        tri_hist.append(pose.tri_ms)
        vic_hist.append(float(getattr(pose.infer, "vic_prep_ms", 0.0)))
        stage_fps = 1000.0 / hist_mean(e2e_hist) if e2e_hist else 0.0
        if frame_idx % cfg.log_every == 0:
            z = z_mean_cm(pose.kpts3d, cfg.conf)
            thr_fps = throughput.rolling_fps()
            win_fps = throughput.interval_fps(frame_idx + 1)
            cap_skip_avg = (
                sum(cap_skip_hist) / len(cap_skip_hist) if cap_skip_hist else 0.0
            )
            prep_skip_avg = (
                sum(prep_skip_hist) / len(prep_skip_hist) if prep_skip_hist else 0.0
            )
            print(
                f"infer={frame_idx} thr_fps={thr_fps:.1f} win_fps={win_fps:.1f} "
                f"stage_fps={stage_fps:.1f} "
                f"seq={pose.infer.capture_seq} Z_mean={z:.1f}cm "
                f"cap={pose.infer.capture_ms:.1f} "
                f"vic={hist_mean(vic_hist):.1f} "
                f"par={pose.infer.infer_parallel_ms:.1f} "
                f"tri={hist_mean(tri_hist):.1f} "
                f"cap_skip={cap_skip_avg:.1f} prep_skip={prep_skip_avg:.1f} "
                f"lb={hist_mean(par_lb_hist):.1f} "
                f"trt={hist_mean(par_trt_hist):.1f} "
                f"post={hist_mean(par_post_hist):.1f} "
                f"drain={hist_mean(par_drain_hist):.1f}",
                flush=True,
            )
        return frame_idx + 1

    packet = build_display_packet(pose, l1=l1, l3=l3)
    e2e_ms = (
        pose.infer.capture_ms
        + pose.infer.infer_parallel_ms
        + pose.tri_ms
    )
    e2e_hist.append(e2e_ms)
    bd = pose.infer.par_breakdown()
    par_egl_hist.append(bd.egl_ms)
    par_lb_hist.append(bd.lb_ms)
    par_trt_hist.append(bd.trt_ms)
    par_post_hist.append(bd.post_ms)
    stage_fps = 1000.0 / hist_mean(e2e_hist) if e2e_hist else 0.0
    thr_fps = throughput.rolling_fps()
    z = z_mean_cm(packet.kpts3d, cfg.conf)
    timing = FrameTiming()
    timing.from_infer_result(pose)
    timing.e2e_ms = e2e_ms
    timing.consumer_ms = packet.timing.consumer_ms
    write_frame_metrics(
        metrics_fp,
        frame_idx,
        timing,
        e2e_fps=thr_fps if thr_fps > 0.0 else stage_fps,
        tri_ok=pose.tri_ok,
        z_mean=round(z, 1),
        capture_seq=pose.infer.capture_seq,
        L_persons=len(pose.infer.left.poses),
        R_persons=len(pose.infer.right.poses),
    )
    if frame_idx % cfg.log_every == 0:
        win_fps = throughput.interval_fps(frame_idx + 1)
        print(
            f"frame={frame_idx} thr_fps={thr_fps:.1f} win_fps={win_fps:.1f} "
            f"stage_fps={stage_fps:.1f} Z_mean={z:.1f}cm "
            f"{timing.format_log_line()}",
            flush=True,
        )
    return frame_idx + 1


def run_infer_worker(
    cfg: Route2SessionConfig,
    pose3d: StereoPose3DPipeline,
    capture_q: queue.Queue,
    stop: threading.Event,
    *,
    prep_q: queue.Queue | None = None,
    prep_ready: threading.Event | None = None,
    pose_store: PoseStore | None = None,
    l1: L1Runtime | None = None,
    l3: L3Runtime | None = None,
    metrics_fp: TextIO | None = None,
    on_frame: Callable[[], None] | None = None,
    c2_auto_stop: C2AutoStopGate | None = None,
) -> None:
    e2e_hist: deque[float] = deque(maxlen=ROLLING_N)
    par_egl_hist: deque[float] = deque(maxlen=ROLLING_N)
    par_lb_hist: deque[float] = deque(maxlen=ROLLING_N)
    par_trt_hist: deque[float] = deque(maxlen=ROLLING_N)
    par_post_hist: deque[float] = deque(maxlen=ROLLING_N)
    par_drain_hist: deque[float] = deque(maxlen=ROLLING_N)
    par_wait_hist: deque[float] = deque(maxlen=ROLLING_N)
    par_unmap_hist: deque[float] = deque(maxlen=ROLLING_N)
    par_trt_sync_hist: deque[float] = deque(maxlen=ROLLING_N)
    tri_hist: deque[float] = deque(maxlen=ROLLING_N)
    vic_hist: deque[float] = deque(maxlen=ROLLING_N)
    cap_skip_hist: deque[int] = deque(maxlen=ROLLING_N)
    prep_skip_hist: deque[int] = deque(maxlen=ROLLING_N)
    throughput = ThroughputTracker()
    frame_idx = 0
    parallel = not cfg.sequential
    split_display = pose_store is not None
    use_vic_prep_q = prep_q is not None and vic_bgr_prep_pipeline_enabled(cfg)
    use_gpu_overlap = (
        cfg.infer_pipeline_overlap
        and split_display
        and cfg.gpu_letterbox
        and not use_vic_prep_q
    )
    staged_item: tuple | None = None

    while not stop.is_set():
        if use_vic_prep_q:
            assert prep_q is not None
            assert prep_ready is not None
            prep_ready.wait(timeout=0.2)
            if stop.is_set():
                break
            got = try_take_latest_from_queue(prep_q)
            if got is None:
                prep_ready.clear()
                continue
            prep_item, prep_q_skipped = got
            prep_skip_hist.append(int(prep_q_skipped))
            cap_skip_hist.append(int(prep_item.capture_skipped))
            try:
                pose = pose3d.process_vic_bgr_prep(
                    prep_item,
                    parallel_eyes=parallel,
                    prep_q_skipped=int(prep_q_skipped),
                )
            except Exception as exc:
                print(f"infer 异常（跳过本帧）: {exc}", flush=True)
                if frame_idx % cfg.log_every == 0:
                    traceback.print_exc()
                continue
        else:
            try:
                if staged_item is not None:
                    item = staged_item
                    staged_item = None
                else:
                    item = capture_q.get(timeout=0.2)
            except queue.Empty:
                continue
            prefetch_sbs = None
            if use_gpu_overlap:
                try:
                    staged_item = capture_q.get_nowait()
                    prefetch_sbs = staged_item[0]
                except queue.Empty:
                    pass
            try:
                sbs, capture_ms = item
                if use_gpu_overlap:
                    pose = pose3d.process_sbs(
                        sbs,
                        capture_ms,
                        parallel_eyes=parallel,
                        release_sbs=True,
                        with_gpu_preview=False,
                        pipeline_overlap=True,
                        prefetch_sbs=prefetch_sbs,
                    )
                    sbs = None
                else:
                    pose = pose3d.process_sbs(
                        sbs,
                        capture_ms,
                        parallel_eyes=parallel,
                        release_sbs=True,
                        with_gpu_preview=False,
                    )
                    sbs = None
            except Exception as exc:
                if isinstance(item, tuple):
                    try:
                        item[0].release()
                    except Exception:
                        pass
                if use_gpu_overlap:
                    try:
                        pose3d.infer_pipeline.reset_overlap_state()
                    except Exception:
                        pass
                    staged_item = None
                print(f"infer 异常（跳过本帧）: {exc}", flush=True)
                if frame_idx % cfg.log_every == 0 or "NoneType" in str(exc):
                    traceback.print_exc()
                continue

        if pose is None:
            continue

        frame_idx = _publish_pose_and_metrics(
            cfg=cfg,
            pose=pose,
            frame_idx=frame_idx,
            throughput=throughput,
            e2e_hist=e2e_hist,
            par_egl_hist=par_egl_hist,
            par_lb_hist=par_lb_hist,
            par_trt_hist=par_trt_hist,
            par_post_hist=par_post_hist,
            par_drain_hist=par_drain_hist,
            par_wait_hist=par_wait_hist,
            par_unmap_hist=par_unmap_hist,
            par_trt_sync_hist=par_trt_sync_hist,
            tri_hist=tri_hist,
            vic_hist=vic_hist,
            cap_skip_hist=cap_skip_hist,
            prep_skip_hist=prep_skip_hist,
            pose_store=pose_store,
            l1=l1,
            l3=l3,
            metrics_fp=metrics_fp,
            on_frame=on_frame,
            c2_auto_stop=c2_auto_stop,
        )
