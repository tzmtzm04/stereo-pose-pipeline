"""JSONL metrics 写入。"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from app.route2.session_config import Route2SessionConfig
from core.geometry.calib_model import StereoCalibration
from orchestration.constants import ROLLING_N
from core.display.pose_store import PoseDisplaySample
from orchestration.types import DisplayPacket, FrameTiming


def apply_pose_infer_timing(
    timing: FrameTiming,
    pose: PoseDisplaySample | None,
) -> None:
    """split_display：UI metrics 从 PoseStore 样本恢复 infer 分项。"""
    if pose is None:
        return
    timing.infer_parallel_ms = float(pose.infer_parallel_ms)
    timing.capture_ms = float(pose.capture_ms)
    timing.infer_egl_ms = float(pose.infer_egl_ms)
    timing.infer_lb_ms = float(pose.infer_lb_ms)
    timing.infer_trt_ms = float(pose.infer_trt_ms)
    timing.infer_post_ms = float(pose.infer_post_ms)
    timing.infer_drain_ms = float(pose.infer_drain_ms)
    timing.infer_wait_ms = float(pose.infer_wait_ms)
    timing.infer_unmap_ms = float(pose.infer_unmap_ms)
    timing.infer_trt_enqueue_ms = float(pose.infer_trt_enqueue_ms)
    timing.infer_trt_d2h_sync_ms = float(pose.infer_trt_d2h_sync_ms)
    timing.triangulate_ms = float(getattr(pose, "tri_ms", 0.0))


def hist_mean(hist: deque[float]) -> float:
    if not hist:
        return 0.0
    return sum(hist) / len(hist)


class ThroughputTracker:
    """墙钟真实吞吐：完成帧时间戳滑动窗口 + 日志区间计数。"""

    def __init__(self, *, window: int = ROLLING_N) -> None:
        self._window = max(2, int(window))
        self._times: deque[float] = deque(maxlen=self._window)
        self._last_log_t: float | None = None
        self._last_log_frames = 0

    def mark(self) -> None:
        self._times.append(time.perf_counter())

    def rolling_fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        dt = self._times[-1] - self._times[0]
        if dt <= 0.0:
            return 0.0
        return (len(self._times) - 1) / dt

    def interval_fps(self, frame_count: int) -> float:
        now = time.perf_counter()
        if self._last_log_t is None:
            self._last_log_t = now
            self._last_log_frames = int(frame_count)
            return 0.0
        df = int(frame_count) - self._last_log_frames
        dt = now - self._last_log_t
        self._last_log_t = now
        self._last_log_frames = int(frame_count)
        if dt <= 0.0 or df <= 0:
            return 0.0
        return df / dt


def z_mean_cm(kpts3d: np.ndarray | None, conf: float) -> float:
    if kpts3d is None:
        return 0.0
    mask = (kpts3d[:, 3] >= conf) & (kpts3d[:, 2] > 0)
    z_vals = kpts3d[mask, 2]
    return float(np.mean(z_vals) * 100) if len(z_vals) else 0.0


def open_metrics_file(
    cfg: Route2SessionConfig,
    calib: StereoCalibration,
    *,
    capture: str,
    mode: str,
) -> TextIO | None:
    if cfg.no_metrics:
        return None
    path = Path(cfg.metrics_log)
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = path.open("w", encoding="utf-8")
    fp.write(
        json.dumps(
            {
                "type": "session",
                "path": "route2_pose3d",
                "mode": mode,
                "calib": str(cfg.calib),
                "baseline_m": calib.baseline_m,
                "engine": str(cfg.engine),
                "capture": capture,
                "display": not cfg.no_display,
                "width": cfg.width,
                "height": cfg.height,
                "fps": cfg.fps,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    return fp


def write_frame_metrics(
    fp: TextIO | None,
    frame_idx: int,
    timing: FrameTiming,
    *,
    e2e_fps: float,
    tri_ok: bool,
    z_mean: float,
    capture_seq: int,
    L_persons: int = 0,
    R_persons: int = 0,
    metrics_stage: str | None = None,
    ui_fps: float | None = None,
) -> None:
    if fp is None:
        return
    row: dict[str, Any] = {
        "type": "frame",
        "frame": frame_idx,
        "e2e_ms": round(timing.e2e_ms, 2),
        "e2e_fps": round(e2e_fps, 2),
        "capture_ms": round(timing.capture_ms, 2),
        "infer_parallel_ms": round(timing.infer_parallel_ms, 2),
        "infer_egl_ms": round(timing.infer_egl_ms, 2),
        "infer_lb_ms": round(timing.infer_lb_ms, 2),
        "infer_trt_ms": round(timing.infer_trt_ms, 2),
        "infer_post_ms": round(timing.infer_post_ms, 2),
        "infer_drain_ms": round(timing.infer_drain_ms, 2),
        "infer_wait_ms": round(timing.infer_wait_ms, 2),
        "infer_unmap_ms": round(timing.infer_unmap_ms, 2),
        "infer_trt_enqueue_ms": round(timing.infer_trt_enqueue_ms, 2),
        "infer_trt_d2h_sync_ms": round(timing.infer_trt_d2h_sync_ms, 2),
        "tri_ms": round(timing.triangulate_ms, 2),
        "preview_ms": round(timing.preview_ms, 2),
        "draw_ms": round(timing.draw_ms, 2),
        "display_ms": round(timing.display_ms, 2),
        "queue_wait_ms": round(timing.queue_wait_ms, 2),
        "tri_3d_ok": tri_ok,
        "Z_mean": z_mean,
        "capture_seq": capture_seq,
        "L_persons": L_persons,
        "R_persons": R_persons,
        "consumer_ms": round(timing.consumer_ms, 2),
    }
    if metrics_stage is not None:
        row["stage"] = metrics_stage
    if ui_fps is not None:
        row["ui_fps"] = round(ui_fps, 1)
    fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    fp.flush()


def commit_packet_metrics(
    fp: TextIO | None,
    packet: DisplayPacket,
    frame_idx: int,
    e2e_hist: deque[float],
    *,
    cfg: Route2SessionConfig,
    queue_wait_ms: float = 0.0,
    display_ms: float = 0.0,
    e2e_ms: float = 0.0,
) -> None:
    timing = packet.timing
    timing.queue_wait_ms = queue_wait_ms
    timing.display_ms = display_ms
    timing.e2e_ms = e2e_ms
    e2e_hist.append(e2e_ms)
    e2e_fps = 1000.0 / hist_mean(e2e_hist) if e2e_hist else 0.0
    z = z_mean_cm(packet.kpts3d, cfg.conf)
    write_frame_metrics(
        fp,
        frame_idx,
        timing,
        e2e_fps=e2e_fps,
        tri_ok=packet.pose.tri_ok,
        z_mean=round(z, 1),
        capture_seq=packet.capture_seq,
        L_persons=len(packet.pose.infer.left.poses),
        R_persons=len(packet.pose.infer.right.poses),
    )
    if frame_idx % cfg.log_every == 0:
        print(
            f"frame={frame_idx} e2e_fps={e2e_fps:.1f} Z_mean={z:.1f}cm "
            f"{timing.format_log_line()}",
            flush=True,
        )
