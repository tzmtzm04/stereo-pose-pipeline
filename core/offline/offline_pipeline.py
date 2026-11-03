"""离线流水线：decode+VIC prep_worker → prep_q → TRT worker。"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Union

import cv2
import numpy as np

from contracts.session_bundle import SessionBundle
from core.offline.offline_pose3d import (
    OfflineBgrPoseInfer,
    OfflineGpuPoseInfer,
    OfflineVicPoseInfer,
    StereoBgrPrep,
    StereoTrtPrep,
)
from core.offline.offline_two_stage_infer import OfflineTwoStageInfer
from core.offline.pose3d_filter_chain import OfflineFilterResult, OfflinePose3DFilterChain
from core.offline.sbs_nv12_vic_split import SbsNv12VicSplitter
from core.offline.session_decoder import SessionIndexRow, split_sbs_bgr
from core.offline.session_nvdec_decoder import SessionNv12Frame, SessionNvdecDecoder
from domain.l1.pose_history import Pose3DSample
from domain.l1.pose_session_log import L1PoseSessionLog


_SENTINEL = object()
_PrepItem = Union[StereoTrtPrep, StereoBgrPrep, Any]


def pose3d_sample_from_filter(
    capture_seq: int,
    infer_seq: int,
    fr: OfflineFilterResult,
) -> Pose3DSample:
    return Pose3DSample(
        capture_seq=int(capture_seq),
        infer_seq=int(infer_seq),
        kpts3d=fr.kpts3d,
        tri_ok=bool(fr.tri_ok),
        det_left=fr.det_left,
        det_right=fr.det_right,
        opencap20_kpts3d=fr.opencap20_kpts3d,
        filter_2d=fr.filter_2d,
    )


@dataclass
class OfflinePipelineStats:
    frames_decoded: int = 0
    frames_pose_written: int = 0
    frames_tri_ok: int = 0
    decode_fail: int = 0
    queue_wait_ms: float = 0.0
    det_q_wait_ms: float = 0.0
    pose_q_wait_ms: float = 0.0
    last_trt_ms: float = 0.0
    last_det_ms: float = 0.0
    last_pose_ms: float = 0.0
    last_prep_ms: float = 0.0
    last_decode_ms: float = 0.0


def _run_decode_vic_prep_worker(
    bundle: SessionBundle,
    rows: list[SessionIndexRow],
    prep_q: queue.Queue,
    stop: threading.Event,
    stats: OfflinePipelineStats,
    stats_lock: threading.Lock,
    *,
    width: int,
    height: int,
    fps: int,
    max_frames: int,
    vic_prep: OfflineVicPoseInfer | None,
    bgr_splitter: SbsNv12VicSplitter | None,
) -> None:
    """NVDEC + VIC（letterbox 或 SBS 裁 BGR）→ prep_q。"""
    decoder = SessionNvdecDecoder(width, height, fps=fps)
    decoded = 0
    try:
        with bundle.mjpeg_path.open("rb") as fp:
            for row in rows:
                if stop.is_set():
                    break
                if max_frames > 0 and decoded >= max_frames:
                    break
                fp.seek(int(row.offset))
                data = fp.read(int(row.size))
                if not data:
                    with stats_lock:
                        stats.decode_fail += 1
                    continue
                t0 = time.perf_counter()
                nv12 = decoder.decode_jpeg(data)
                decode_ms = (time.perf_counter() - t0) * 1000.0
                if nv12 is None:
                    with stats_lock:
                        stats.decode_fail += 1
                    continue
                try:
                    if vic_prep is not None:
                        item: _PrepItem = vic_prep.prep_sbs_nv12(
                            nv12,
                            int(row.capture_seq),
                            decode_ms,
                        )
                        prep_ms = float(item.prep_ms)
                    else:
                        assert bgr_splitter is not None
                        t1 = time.perf_counter()
                        left, right = bgr_splitter.split_eye_bgr(nv12)
                        prep_ms = (time.perf_counter() - t1) * 1000.0
                        item = StereoBgrPrep(
                            capture_seq=int(row.capture_seq),
                            left_bgr=left,
                            right_bgr=right,
                            decode_ms=decode_ms,
                            prep_ms=prep_ms,
                        )
                finally:
                    nv12.release()
                while not stop.is_set():
                    try:
                        prep_q.put(item, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    if isinstance(item, StereoTrtPrep):
                        item.release_vic()
                    break
                decoded += 1
                with stats_lock:
                    stats.frames_decoded += 1
                    stats.last_decode_ms = decode_ms
                    stats.last_prep_ms = prep_ms
    finally:
        decoder.close()
        prep_q.put(_SENTINEL)


def _run_decode_nvdec_only_worker(
    bundle: SessionBundle,
    rows: list[SessionIndexRow],
    prep_q: queue.Queue,
    stop: threading.Event,
    stats: OfflinePipelineStats,
    stats_lock: threading.Lock,
    *,
    width: int,
    height: int,
    fps: int,
    max_frames: int,
    appsink_max_buffers: int = 1,
) -> None:
    """NVDEC only（--gpu-letterbox：CUDA prep 在 TRT 线程）。"""
    decoder = SessionNvdecDecoder(
        width, height, fps=fps, appsink_max_buffers=int(appsink_max_buffers)
    )
    decoded = 0
    try:
        with bundle.mjpeg_path.open("rb") as fp:
            for row in rows:
                if stop.is_set():
                    break
                if max_frames > 0 and decoded >= max_frames:
                    break
                fp.seek(int(row.offset))
                data = fp.read(int(row.size))
                if not data:
                    with stats_lock:
                        stats.decode_fail += 1
                    continue
                t0 = time.perf_counter()
                nv12 = decoder.decode_jpeg(data)
                decode_ms = (time.perf_counter() - t0) * 1000.0
                if nv12 is None:
                    with stats_lock:
                        stats.decode_fail += 1
                    continue
                item = SessionNv12Frame(
                    capture_seq=int(row.capture_seq),
                    sbs_nv12=nv12,
                )
                wrapper = (item, decode_ms)
                while not stop.is_set():
                    try:
                        prep_q.put(wrapper, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                if stop.is_set():
                    item.release()
                    break
                decoded += 1
                with stats_lock:
                    stats.frames_decoded += 1
                    stats.last_decode_ms = decode_ms
    finally:
        decoder.close()
        prep_q.put(_SENTINEL)


def _run_decode_sw_worker(
    bundle: SessionBundle,
    rows: list[SessionIndexRow],
    prep_q: queue.Queue,
    stop: threading.Event,
    stats: OfflinePipelineStats,
    stats_lock: threading.Lock,
    *,
    max_frames: int,
) -> None:
    decoded = 0
    with bundle.mjpeg_path.open("rb") as fp:
        for row in rows:
            if stop.is_set():
                break
            if max_frames > 0 and decoded >= max_frames:
                break
            fp.seek(int(row.offset))
            data = fp.read(int(row.size))
            if not data:
                with stats_lock:
                    stats.decode_fail += 1
                continue
            t0 = time.perf_counter()
            arr = np.frombuffer(data, dtype=np.uint8)
            sbs = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            decode_ms = (time.perf_counter() - t0) * 1000.0
            if sbs is None:
                with stats_lock:
                    stats.decode_fail += 1
                continue
            left, right = split_sbs_bgr(sbs)
            item = StereoBgrPrep(
                capture_seq=int(row.capture_seq),
                left_bgr=left,
                right_bgr=right,
                decode_ms=decode_ms,
                prep_ms=0.0,
            )
            while not stop.is_set():
                try:
                    prep_q.put(item, timeout=0.1)
                    break
                except queue.Full:
                    continue
            if stop.is_set():
                break
            decoded += 1
            with stats_lock:
                stats.frames_decoded += 1
                stats.last_decode_ms = decode_ms
    prep_q.put(_SENTINEL)


def _run_trt_worker_vic(
    infer: OfflineVicPoseInfer,
    filter_chain: OfflinePose3DFilterChain,
    pose_log: L1PoseSessionLog,
    prep_q: queue.Queue,
    stats: OfflinePipelineStats,
    stats_lock: threading.Lock,
    *,
    log_every: int,
    infer_path: str,
    decode_path: str,
    filter_mode: str,
    t0: float,
) -> None:
    infer_seq = 0
    while True:
        t_wait = time.perf_counter()
        item = prep_q.get()
        queue_wait_ms = (time.perf_counter() - t_wait) * 1000.0
        if item is _SENTINEL:
            prep_q.put(_SENTINEL)
            break
        assert isinstance(item, StereoTrtPrep)
        try:
            det_l, det_r, trt_ms = infer.infer_prepared(item)
            infer_seq += 1
            fr = filter_chain.process_pair_ex(
                det_l,
                det_r,
                capture_seq=int(item.capture_seq),
                infer_seq=infer_seq,
            )
            pose_log.append_sample(
                pose3d_sample_from_filter(int(item.capture_seq), infer_seq, fr)
            )
            tri_ok = fr.tri_ok
            det_l = fr.det_left
            with stats_lock:
                stats.frames_pose_written += 1
                stats.queue_wait_ms = queue_wait_ms
                stats.last_trt_ms = trt_ms
                if tri_ok:
                    stats.frames_tri_ok += 1
                n = stats.frames_pose_written
                tri_i = int(tri_ok)
            if log_every > 0 and n % log_every == 0:
                elapsed = time.perf_counter() - t0
                fps_wall = n / max(elapsed, 1e-6)
                print(
                    f"offline seq={item.capture_seq} n={n} tri_ok={tri_i} "
                    f"trt_ms={trt_ms:.1f} prep_ms={item.prep_ms:.1f} "
                    f"decode_ms={item.decode_ms:.1f} q_wait_ms={queue_wait_ms:.1f} "
                    f"wall_fps={fps_wall:.2f} decode={decode_path} infer={infer_path} "
                    f"filter={filter_mode}",
                    flush=True,
                )
        finally:
            item.release_vic()


def _run_trt_worker_gpu(
    infer: OfflineGpuPoseInfer,
    filter_chain: OfflinePose3DFilterChain,
    pose_log: L1PoseSessionLog,
    prep_q: queue.Queue,
    stats: OfflinePipelineStats,
    stats_lock: threading.Lock,
    *,
    log_every: int,
    infer_path: str,
    decode_path: str,
    filter_mode: str,
    t0: float,
) -> None:
    infer_seq = 0
    while True:
        t_wait = time.perf_counter()
        item = prep_q.get()
        queue_wait_ms = (time.perf_counter() - t_wait) * 1000.0
        if item is _SENTINEL:
            prep_q.put(_SENTINEL)
            break
        frame, decode_ms = item
        try:
            if isinstance(infer, OfflineTwoStageInfer):
                det_l, det_r, _block_ms = infer.infer_det_then_pose_sbs(
                    frame.sbs_nv12,
                    capture_seq=int(frame.capture_seq),
                    decode_ms=float(decode_ms),
                )
            else:
                det_l, det_r, _block_ms = infer.infer_stereo_sbs(frame.sbs_nv12)
            prep_ms = float(infer.last_prep_ms)
            trt_ms = float(infer.last_trt_ms)
            infer_seq += 1
            fr = filter_chain.process_pair_ex(
                det_l,
                det_r,
                capture_seq=int(frame.capture_seq),
                infer_seq=infer_seq,
            )
            pose_log.append_sample(
                pose3d_sample_from_filter(int(frame.capture_seq), infer_seq, fr)
            )
            tri_ok = fr.tri_ok
            det_l = fr.det_left
            with stats_lock:
                stats.frames_pose_written += 1
                stats.queue_wait_ms = queue_wait_ms
                stats.last_trt_ms = trt_ms
                stats.last_prep_ms = prep_ms
                if tri_ok:
                    stats.frames_tri_ok += 1
                n = stats.frames_pose_written
                tri_i = int(tri_ok)
            if log_every > 0 and n % log_every == 0:
                elapsed = time.perf_counter() - t0
                fps_wall = n / max(elapsed, 1e-6)
                print(
                    f"offline seq={frame.capture_seq} n={n} tri_ok={tri_i} "
                    f"trt_ms={trt_ms:.1f} prep_ms={prep_ms:.1f} "
                    f"decode_ms={decode_ms:.1f} q_wait_ms={queue_wait_ms:.1f} "
                    f"wall_fps={fps_wall:.2f} decode={decode_path} "
                    f"infer={infer_path} filter={filter_mode}",
                    flush=True,
                )
        finally:
            frame.release()


def _run_trt_worker_cpu(
    infer: OfflineBgrPoseInfer | OfflineTwoStageInfer,
    filter_chain: OfflinePose3DFilterChain,
    pose_log: L1PoseSessionLog,
    prep_q: queue.Queue,
    stats: OfflinePipelineStats,
    stats_lock: threading.Lock,
    *,
    log_every: int,
    infer_path: str,
    decode_path: str,
    filter_mode: str,
    t0: float,
) -> None:
    two_stage = isinstance(infer, OfflineTwoStageInfer)
    infer_seq = 0
    while True:
        t_wait = time.perf_counter()
        item = prep_q.get()
        queue_wait_ms = (time.perf_counter() - t_wait) * 1000.0
        if item is _SENTINEL:
            prep_q.put(_SENTINEL)
            break
        assert isinstance(item, StereoBgrPrep)
        t_trt = time.perf_counter()
        det_l, det_r, infer_ms = infer.infer_stereo(item.left_bgr, item.right_bgr)
        trt_ms = (time.perf_counter() - t_trt) * 1000.0
        det_ms = float(infer.last_det_ms) if two_stage else 0.0
        pose_ms = float(infer.last_pose_ms) if two_stage else 0.0
        infer_seq += 1
        fr = filter_chain.process_pair_ex(
            det_l,
            det_r,
            capture_seq=int(item.capture_seq),
            infer_seq=infer_seq,
        )
        pose_log.append_sample(
            pose3d_sample_from_filter(int(item.capture_seq), infer_seq, fr)
        )
        tri_ok = fr.tri_ok
        with stats_lock:
            stats.frames_pose_written += 1
            stats.queue_wait_ms = queue_wait_ms
            stats.last_trt_ms = trt_ms
            if two_stage:
                stats.last_det_ms = det_ms
                stats.last_pose_ms = pose_ms
                stats.last_prep_ms = float(item.prep_ms)
            if tri_ok:
                stats.frames_tri_ok += 1
            n = stats.frames_pose_written
            tri_i = int(tri_ok)
        if log_every > 0 and n % log_every == 0:
            elapsed = time.perf_counter() - t0
            fps_wall = n / max(elapsed, 1e-6)
            if two_stage:
                print(
                    f"offline seq={item.capture_seq} n={n} tri_ok={tri_i} "
                    f"det_ms={det_ms:.1f} pose_ms={pose_ms:.1f} "
                    f"prep_ms={item.prep_ms:.1f} decode_ms={item.decode_ms:.1f} "
                    f"q_wait_ms={queue_wait_ms:.1f} "
                    f"wall_fps={fps_wall:.2f} decode={decode_path} "
                    f"infer={infer_path} filter={filter_mode}",
                    flush=True,
                )
            else:
                print(
                    f"offline seq={item.capture_seq} n={n} tri_ok={tri_i} "
                    f"trt_ms={trt_ms:.1f} prep_ms={item.prep_ms:.1f} "
                    f"decode_ms={item.decode_ms:.1f} q_wait_ms={queue_wait_ms:.1f} "
                    f"wall_fps={fps_wall:.2f} decode={decode_path} infer={infer_path} "
                    f"filter={filter_mode}",
                    flush=True,
                )
        _ = infer_ms


def _run_two_stage_serial_nvdec_worker(
    bundle: SessionBundle,
    rows: list[SessionIndexRow],
    infer: OfflineTwoStageInfer,
    filter_chain: OfflinePose3DFilterChain,
    pose_log: L1PoseSessionLog,
    stats: OfflinePipelineStats,
    stats_lock: threading.Lock,
    *,
    width: int,
    height: int,
    fps: int,
    max_frames: int,
    log_every: int,
    infer_path: str,
    decode_path: str,
    filter_mode: str,
    t0: float,
) -> None:
    """NVDEC 串行：decode → VIC BGR → infer（与 imdecode 像素对齐）。"""
    decoder = SessionNvdecDecoder(width, height, fps=fps, appsink_max_buffers=1)
    infer_seq = 0
    decoded = 0
    try:
        with bundle.mjpeg_path.open("rb") as fp:
            for row in rows:
                if max_frames > 0 and decoded >= max_frames:
                    break
                fp.seek(int(row.offset))
                data = fp.read(int(row.size))
                if not data:
                    with stats_lock:
                        stats.decode_fail += 1
                    continue
                t_dec = time.perf_counter()
                nv12 = decoder.decode_jpeg(data)
                decode_ms = (time.perf_counter() - t_dec) * 1000.0
                if nv12 is None:
                    with stats_lock:
                        stats.decode_fail += 1
                    continue
                try:
                    det_l, det_r, block_ms = infer.infer_stereo_nv12_vic_bgr(
                        nv12,
                        capture_seq=int(row.capture_seq),
                        decode_ms=float(decode_ms),
                    )
                    prep_ms = float(infer.last_prep_ms)
                    det_ms = float(infer.last_det_ms)
                    pose_ms = float(infer.last_pose_ms)
                    infer_seq += 1
                    fr = filter_chain.process_pair_ex(
                        det_l,
                        det_r,
                        capture_seq=int(row.capture_seq),
                        infer_seq=infer_seq,
                    )
                    pose_log.append_sample(
                        pose3d_sample_from_filter(int(row.capture_seq), infer_seq, fr)
                    )
                    tri_ok = fr.tri_ok
                    decoded += 1
                    with stats_lock:
                        stats.frames_decoded += 1
                        stats.frames_pose_written += 1
                        stats.last_decode_ms = decode_ms
                        stats.last_prep_ms = prep_ms
                        stats.last_det_ms = det_ms
                        stats.last_pose_ms = pose_ms
                        stats.last_trt_ms = float(block_ms)
                        if tri_ok:
                            stats.frames_tri_ok += 1
                        n = stats.frames_pose_written
                        tri_i = int(tri_ok)
                    if log_every > 0 and n % log_every == 0:
                        elapsed = time.perf_counter() - t0
                        fps_wall = n / max(elapsed, 1e-6)
                        print(
                            f"offline seq={row.capture_seq} n={n} tri_ok={tri_i} "
                            f"det_ms={det_ms:.1f} pose_ms={pose_ms:.1f} "
                            f"prep_ms={prep_ms:.1f} decode_ms={decode_ms:.1f} "
                            f"wall_fps={fps_wall:.2f} decode={decode_path} "
                            f"infer={infer_path} filter={filter_mode}",
                            flush=True,
                        )
                finally:
                    nv12.release()
    finally:
        decoder.close()


def _run_two_stage_infer_worker(
    infer: OfflineTwoStageInfer,
    filter_chain: OfflinePose3DFilterChain,
    pose_log: L1PoseSessionLog,
    decode_q: queue.Queue,
    stats: OfflinePipelineStats,
    stats_lock: threading.Lock,
    *,
    log_every: int,
    infer_path: str,
    decode_path: str,
    filter_mode: str,
    t0: float,
) -> None:
    """decode_q → 同帧 det→pose 原子 infer → 滤波写盘。"""
    infer_seq = 0
    while True:
        t_wait = time.perf_counter()
        item = decode_q.get()
        queue_wait_ms = (time.perf_counter() - t_wait) * 1000.0
        if item is _SENTINEL:
            decode_q.put(_SENTINEL)
            break
        frame, decode_ms = item
        try:
            det_l, det_r, block_ms = infer.infer_det_then_pose_sbs(
                frame.sbs_nv12,
                capture_seq=int(frame.capture_seq),
                decode_ms=float(decode_ms),
            )
            prep_ms = float(infer.last_prep_ms)
            det_ms = float(infer.last_det_ms)
            pose_ms = float(infer.last_pose_ms)
            infer_seq += 1
            fr = filter_chain.process_pair_ex(
                det_l,
                det_r,
                capture_seq=int(frame.capture_seq),
                infer_seq=infer_seq,
            )
            pose_log.append_sample(
                pose3d_sample_from_filter(int(frame.capture_seq), infer_seq, fr)
            )
            tri_ok = fr.tri_ok
            with stats_lock:
                stats.frames_pose_written += 1
                stats.queue_wait_ms = queue_wait_ms
                stats.last_prep_ms = prep_ms
                stats.last_det_ms = det_ms
                stats.last_pose_ms = pose_ms
                stats.last_trt_ms = float(block_ms)
                if tri_ok:
                    stats.frames_tri_ok += 1
                n = stats.frames_pose_written
                tri_i = int(tri_ok)
            if log_every > 0 and n % log_every == 0:
                elapsed = time.perf_counter() - t0
                fps_wall = n / max(elapsed, 1e-6)
                print(
                    f"offline seq={frame.capture_seq} n={n} tri_ok={tri_i} "
                    f"det_ms={det_ms:.1f} pose_ms={pose_ms:.1f} "
                    f"prep_ms={prep_ms:.1f} decode_ms={decode_ms:.1f} "
                    f"q_wait_ms={queue_wait_ms:.1f} "
                    f"wall_fps={fps_wall:.2f} decode={decode_path} "
                    f"infer={infer_path} filter={filter_mode}",
                    flush=True,
                )
        finally:
            frame.release()


def run_offline_pipeline(
    bundle: SessionBundle,
    rows: list[SessionIndexRow],
    infer: OfflineVicPoseInfer | OfflineGpuPoseInfer | OfflineBgrPoseInfer | OfflineTwoStageInfer,
    filter_chain: OfflinePose3DFilterChain,
    pose_log: L1PoseSessionLog,
    *,
    use_nvdec: bool,
    use_hw_infer: bool,
    gpu_letterbox: bool,
    infer_path: str,
    decode_path: str,
    filter_mode: str,
    queue_depth: int = 2,
    max_frames: int = 0,
    log_every: int = 30,
    width: int = 3840,
    height: int = 1080,
    fps: int = 60,
    eye_w: int = 1920,
    eye_h: int = 1080,
) -> OfflinePipelineStats:
    """离线流水线：decode 线程 prep → prep_q → infer 线程（two_stage VIC BGR 亦并行）。"""
    _ = gpu_letterbox
    pipe_q = max(2, int(queue_depth))
    prep_q: queue.Queue = queue.Queue(maxsize=pipe_q)
    stop = threading.Event()
    stats = OfflinePipelineStats()
    stats_lock = threading.Lock()
    t0 = time.perf_counter()

    two_stage_vic = isinstance(infer, OfflineTwoStageInfer) and infer.vic_bgr_cpu_path
    two_stage_vic_geom = isinstance(infer, OfflineTwoStageInfer) and infer.vic_geom_gpu_path
    two_stage_gpu = isinstance(infer, OfflineTwoStageInfer) and infer.gpu_mode

    bgr_splitter: SbsNv12VicSplitter | None = None
    decode_target = _run_decode_sw_worker
    decode_kwargs: dict[str, Any] = {}

    if use_nvdec and use_hw_infer and isinstance(infer, OfflineVicPoseInfer):
        decode_target = _run_decode_vic_prep_worker
        decode_kwargs = {
            "width": width,
            "height": height,
            "fps": fps,
            "vic_prep": infer,
            "bgr_splitter": None,
        }
    elif use_nvdec and (two_stage_vic or not use_hw_infer):
        bgr_splitter = SbsNv12VicSplitter(eye_w, eye_h)
        decode_target = _run_decode_vic_prep_worker
        decode_kwargs = {
            "width": width,
            "height": height,
            "fps": fps,
            "vic_prep": None,
            "bgr_splitter": bgr_splitter,
        }
    elif use_nvdec and (isinstance(infer, OfflineGpuPoseInfer) or two_stage_vic_geom or two_stage_gpu):
        decode_target = _run_decode_nvdec_only_worker
        decode_kwargs = {"width": width, "height": height, "fps": fps}

    if isinstance(infer, OfflineVicPoseInfer):
        infer_target = _run_trt_worker_vic
    elif isinstance(infer, OfflineGpuPoseInfer) or two_stage_vic_geom or two_stage_gpu:
        infer_target = _run_trt_worker_gpu
    else:
        infer_target = _run_trt_worker_cpu

    infer_kwargs: dict[str, Any] = {}

    decode_th = threading.Thread(
        target=decode_target,
        kwargs={
            "bundle": bundle,
            "rows": rows,
            "prep_q": prep_q,
            "stop": stop,
            "stats": stats,
            "stats_lock": stats_lock,
            "max_frames": max_frames,
            **decode_kwargs,
        },
        name="offline_decode_vic_prep",
        daemon=True,
    )
    infer_th = threading.Thread(
        target=infer_target,
        kwargs={
            "infer": infer,
            "filter_chain": filter_chain,
            "pose_log": pose_log,
            "prep_q": prep_q,
            "stats": stats,
            "stats_lock": stats_lock,
            "log_every": log_every,
            "infer_path": infer_path,
            "decode_path": decode_path,
            "filter_mode": filter_mode,
            "t0": t0,
            **infer_kwargs,
        },
        name="offline_trt",
        daemon=True,
    )
    decode_th.start()
    infer_th.start()
    decode_th.join()
    infer_th.join()
    if bgr_splitter is not None:
        bgr_splitter.close()
    return stats
