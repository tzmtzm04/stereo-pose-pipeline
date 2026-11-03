"""流水线：capture → infer → split display（或 no-display）。"""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any, TextIO

from app.route2.session_config import Route2SessionConfig
from core.display.pose_store import PoseStore
from core.infer.pose3d_pipeline import StereoPose3DPipeline, StereoPose3DResult
from domain.l1.runtime import L1Runtime
from domain.l3.runtime import L3Runtime
from orchestration.workers.capture_worker import run_capture_worker
from orchestration.workers.display_worker import run_split_display_worker
from orchestration.workers.infer_worker import run_infer_worker
from orchestration.workers.vic_bgr_prep_worker import run_vic_bgr_prep_worker
from orchestration.pose_publish import publish_infer_pose
from orchestration.vic_bgr_pipeline import vic_bgr_prep_pipeline_enabled
from orchestration.worker_join import join_pipeline_workers
from platform.jetson.capture import JetsonGstNvmmCapture


from app.sdk._bind import bind_embedded_realtime


def run_pipeline_loop(
    cfg: Route2SessionConfig,
    cap: JetsonGstNvmmCapture,
    pose3d: StereoPose3DPipeline,
    metrics_fp: TextIO | None,
    first_pose: StereoPose3DResult,
    *,
    l1: L1Runtime | None = None,
    l3: L3Runtime | None = None,
    grab_runtime: Any | None = None,
    session_dir: Path | None = None,
) -> None:
    pipe_q = max(2, int(cfg.pipeline_queue_depth))
    capture_q: queue.Queue = queue.Queue(maxsize=pipe_q)
    prep_q: queue.Queue | None = None
    prep_ready: threading.Event | None = None
    use_vic_prep = vic_bgr_prep_pipeline_enabled(cfg)
    if use_vic_prep:
        prep_q = queue.Queue(maxsize=max(2, pipe_q))
        prep_ready = threading.Event()
    pose_store: PoseStore | None = None
    if cfg.split_display:
        pose_store = PoseStore(max_samples=max(512, cfg.fps * 60))
    stop = threading.Event()
    quit_evt = threading.Event()
    c2_gate = None
    if cfg.c2_auto_stop and cfg.split_display:
        try:
            from orchestration.c2_auto_stop import C2AutoStopGate
        except ImportError:
            C2AutoStopGate = None  # type: ignore[misc, assignment]
        if C2AutoStopGate is not None:
            c2_gate = C2AutoStopGate(
                fps=float(cfg.fps),
                session_dir=session_dir,
                stop=stop,
                quit_evt=quit_evt,
            )
    max_frames = max(0, int(cfg.max_frames))
    frames_done = {"n": 0}
    frames_lock = threading.Lock()

    def _maybe_stop_after_frame() -> None:
        if max_frames <= 0:
            return
        with frames_lock:
            frames_done["n"] += 1
            if frames_done["n"] >= max_frames:
                stop.set()

    capture_th = threading.Thread(
        target=run_capture_worker,
        kwargs={
            "cfg": cfg,
            "cap": cap,
            "capture_q": capture_q,
            "stop": stop,
            "grab_runtime": grab_runtime,
        },
        name="route2_capture",
        daemon=True,
    )
    capture_th.start()

    prep_th: threading.Thread | None = None
    if use_vic_prep and prep_q is not None:
        prep_th = threading.Thread(
            target=run_vic_bgr_prep_worker,
            kwargs={
                "cfg": cfg,
                "infer_pipeline": pose3d.infer_pipeline,
                "capture_q": capture_q,
                "prep_q": prep_q,
                "prep_ready": prep_ready,
                "stop": stop,
            },
            name="route2_vic_bgr_prep",
            daemon=True,
        )
        prep_th.start()

    infer_th = threading.Thread(
        target=run_infer_worker,
        kwargs={
            "cfg": cfg,
            "pose3d": pose3d,
            "capture_q": capture_q,
            "stop": stop,
            "prep_q": prep_q,
            "prep_ready": prep_ready,
            "pose_store": pose_store,
            "l1": l1,
            "l3": l3,
            "metrics_fp": metrics_fp if cfg.no_display else None,
            "on_frame": _maybe_stop_after_frame,
            "c2_auto_stop": c2_gate,
        },
        name="route2_infer_consumer",
        daemon=True,
    )
    infer_th.start()

    def _join_workers() -> None:
        workers: list[tuple[str, threading.Thread]] = [
            ("capture", capture_th),
            ("infer", infer_th),
        ]
        if prep_th is not None:
            workers.insert(1, ("prep", prep_th))
        join_pipeline_workers(workers, stop=stop)

    if cfg.no_display:
        while not stop.is_set():
            time.sleep(0.05)
        _join_workers()
        return

    if cap.sink_display_left is None:
        raise RuntimeError("split_display 需要 sink_display_left（display tee 支路）")
    assert pose_store is not None
    bind_embedded_realtime(cfg, pose_store, stop, quit_evt)
    publish_infer_pose(
        first_pose,
        pose_store,
        l1=l1,
        l3=l3,
        c2_auto_stop=c2_gate,
    )

    run_split_display_worker(
        cfg,
        cap,
        pose_store,
        stop,
        quit_evt,
        metrics_fp,
        l3=l3,
    )
    if not stop.is_set():
        stop.set()
    _join_workers()
