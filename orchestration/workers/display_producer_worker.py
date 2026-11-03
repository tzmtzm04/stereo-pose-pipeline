"""Display 出图线：pull NV12 → slot（策略 D + P4 EGL/GL 或 BGR）。"""

from __future__ import annotations

import threading
import time
import traceback
from typing import Any

from app.route2.session_config import Route2SessionConfig
from core.display.display_tick import DisplayFrameSlot, DisplayTick
from core.display.video_ring import DisplayVideoRing
from core.display.pose_store import PoseDisplaySample, PoseStore
from core.display.gpu_display_preview import _is_sbs_full_frame
from core.display.sbs_preview import display_preview_shape, nv12_display_frame_to_preview_bgr
from core.display.vic_display_bgr import DisplayVicEyeBgr
from orchestration.metrics import ThroughputTracker
from platform.jetson.capture import JetsonGstNvmmCapture

DISPLAY_PULL_TIMEOUT_SEC = 0.35
POSE_WAIT_TIMEOUT_SEC = 0.5
PROD_PULL_FAIL_LOG_EVERY = 30


def run_display_producer_worker(
    cfg: Route2SessionConfig,
    cap: JetsonGstNvmmCapture,
    store: PoseStore,
    frame_slot: DisplayFrameSlot,
    stop: threading.Event,
    *,
    wait_pose: bool = False,
    publish_nv12: bool | None = None,
    video_ring: DisplayVideoRing | None = None,
) -> None:
    """默认策略 D：持续出 seq+bgr 或 NV12 fd；wait_pose 仅 legacy 调试。"""
    use_nv12 = (
        bool(cfg.display_use_egl_gl)
        if publish_nv12 is None
        else bool(publish_nv12)
    )
    throughput = ThroughputTracker()
    vic: DisplayVicEyeBgr | None = None
    if cfg.display_vic_bgr_decode:
        vic = DisplayVicEyeBgr(
            eye_w=cfg.eye_width,
            eye_h=cfg.height,
            eye_rotate="none",
        )
    vic_tag = " VIC→BGR（同 infer 解码）" if vic is not None else ""
    print(
        "display producer 线程启动"
        + ("（legacy 等 pose）" if wait_pose else "（策略 D 纯视频）")
        + (" EGL/GL" if use_nv12 else "")
        + vic_tag,
        flush=True,
    )
    try:
        if wait_pose:
            _produce_wait_pose(
                cfg, cap, store, frame_slot, stop, throughput, use_nv12=use_nv12, vic=vic
            )
        else:
            _produce_video(
                cfg,
                cap,
                frame_slot,
                stop,
                throughput,
                use_nv12=use_nv12,
                video_ring=video_ring,
                vic=vic,
            )
    except Exception as exc:
        print(f"display producer 异常退出: {exc}", flush=True)
        traceback.print_exc()
    finally:
        if vic is not None:
            vic.close()
        frame_slot.close()


def _produce_video(
    cfg: Route2SessionConfig,
    cap: JetsonGstNvmmCapture,
    frame_slot: DisplayFrameSlot,
    stop: threading.Event,
    throughput: ThroughputTracker,
    *,
    use_nv12: bool,
    video_ring: DisplayVideoRing | None = None,
    vic: DisplayVicEyeBgr | None = None,
) -> None:
    """持续 pull display NV12 → BGR，只带 capture_seq，不查 PoseStore。"""
    prod_idx = 0

    while not stop.is_set():
        cycle_start = time.perf_counter()
        t_pull = time.perf_counter()
        ok, frame, capture_seq = cap.read_display_nv12()
        pull_ms = (time.perf_counter() - t_pull) * 1000.0
        if not ok or frame is None:
            fails = _record_prod_pull_fail(frame_slot, video_ring)
            if fails == 1 or fails % PROD_PULL_FAIL_LOG_EVERY == 0:
                print(
                    f"prod pull_fail streak={fails} pull_ms={pull_ms:.1f}",
                    flush=True,
                )
            if stop.is_set():
                break
            time.sleep(0.001)
            continue
        # GL：tick 持有 nv12，由 UI render 后 release（transferred=True）
        # CPU：BGR 已拷走，必须在此释放 nv12，否则耗尽 NVMM 池 → pull/infer 全死
        transferred = False
        try:
            if use_nv12:
                tick = _nv12_to_slot_tick_gl(frame, capture_seq, pull_ms=pull_ms)
                if tick is not None:
                    prod_idx = _emit_prod_tick(
                        cfg,
                        frame_slot,
                        tick,
                        prod_idx,
                        throughput,
                        cycle_start,
                        capture_seq=int(capture_seq),
                        video_ring=video_ring,
                    )
                    transferred = True
                else:
                    print(
                        f"prod nv12→gl 失败 seq={capture_seq} pull={pull_ms:.1f}",
                        flush=True,
                    )
            else:
                tick = _nv12_to_slot_tick(
                    cfg, frame_slot, frame, capture_seq, pull_ms=pull_ms, vic=vic
                )
                if tick is not None:
                    prod_idx = _emit_prod_tick(
                        cfg,
                        frame_slot,
                        tick,
                        prod_idx,
                        throughput,
                        cycle_start,
                        capture_seq=int(capture_seq),
                        video_ring=video_ring,
                    )
                else:
                    print(
                        f"prod nv12→bgr 失败 seq={capture_seq} pull={pull_ms:.1f}",
                        flush=True,
                    )
        finally:
            if not transferred:
                try:
                    frame.release()
                except Exception:
                    pass


def _produce_wait_pose(
    cfg: Route2SessionConfig,
    cap: JetsonGstNvmmCapture,
    store: PoseStore,
    frame_slot: DisplayFrameSlot,
    stop: threading.Event,
    throughput: ThroughputTracker,
    *,
    use_nv12: bool,
    vic: DisplayVicEyeBgr | None = None,
) -> None:
    """Legacy：等 infer pose 再 pull 对齐帧（拖慢 prod，非默认）。"""
    last_pose_seq = -1
    prod_idx = 0

    while not stop.is_set():
        cycle_start = time.perf_counter()
        t_wait = time.perf_counter()
        pose = store.wait_for_pose_after(
            last_pose_seq,
            timeout=POSE_WAIT_TIMEOUT_SEC,
        )
        wait_ms = (time.perf_counter() - t_wait) * 1000.0
        if pose is None:
            if stop.is_set():
                break
            continue

        target_seq = int(pose.capture_seq)
        last_pose_seq = target_seq

        t_pull = time.perf_counter()
        pulled = _pull_display_frame_for_pose(
            cap,
            store,
            target_seq,
            timeout_sec=DISPLAY_PULL_TIMEOUT_SEC,
        )
        pull_ms = (time.perf_counter() - t_pull) * 1000.0
        if pulled is None:
            fails = frame_slot.record_pull_fail()
            if fails == 1 or fails % PROD_PULL_FAIL_LOG_EVERY == 0:
                print(
                    f"prod pull_fail streak={fails} mode=wait_pose "
                    f"target_seq={target_seq}",
                    flush=True,
                )
            continue
        frame, capture_seq, _pose = pulled
        transferred = False
        try:
            if use_nv12:
                tick = _nv12_to_slot_tick_gl(
                    frame, capture_seq, pull_ms=pull_ms, wait_ms=wait_ms
                )
            else:
                tick = _nv12_to_slot_tick(
                    cfg,
                    frame_slot,
                    frame,
                    capture_seq,
                    pull_ms=pull_ms,
                    wait_ms=wait_ms,
                    vic=vic,
                )
            if tick is not None:
                transferred = use_nv12
                prod_idx = _emit_prod_tick(
                    cfg,
                    frame_slot,
                    tick,
                    prod_idx,
                    throughput,
                    cycle_start,
                    capture_seq=capture_seq,
                )
        finally:
            if not transferred:
                try:
                    frame.release()
                except Exception:
                    pass


def _record_prod_pull_fail(
    frame_slot: DisplayFrameSlot,
    video_ring: DisplayVideoRing | None,
) -> int:
    if video_ring is not None:
        return video_ring.record_pull_fail()
    return frame_slot.record_pull_fail()


def _emit_prod_tick(
    cfg: Route2SessionConfig,
    frame_slot: DisplayFrameSlot,
    tick: DisplayTick,
    prod_idx: int,
    throughput: ThroughputTracker,
    cycle_start: float,
    *,
    capture_seq: int,
    video_ring: DisplayVideoRing | None = None,
) -> int:
    if video_ring is not None:
        video_ring.push(tick)
    else:
        frame_slot.publish_tick(tick)
    throughput.mark()
    prod_ms = (time.perf_counter() - cycle_start) * 1000.0
    if prod_idx == 0:
        if tick.nv12 is not None:
            bgr_path = "EGL/GL"
        elif cfg.display_vic_bgr_decode:
            bgr_path = "VIC"
        elif cfg.cpu_preview:
            bgr_path = "CPU"
        else:
            bgr_path = "GPU"
        print(
            f"prod 首帧 OK seq={capture_seq} path={bgr_path} "
            f"pull={tick.pull_ms:.1f} prev={tick.preview_ms:.1f}",
            flush=True,
        )
    if prod_idx % cfg.log_every == 0:
        prod_fps = throughput.rolling_fps()
        win_fps = throughput.interval_fps(prod_idx + 1)
        wait_tag = f" wait={tick.wait_ms:.1f}" if tick.wait_ms > 0 else ""
        print(
            f"prod={prod_idx} prod_fps={prod_fps:.1f} win_fps={win_fps:.1f} "
            f"seq={capture_seq}{wait_tag} "
            f"pull={tick.pull_ms:.1f} prev={tick.preview_ms:.1f} "
            f"prod_ms={prod_ms:.1f}",
            flush=True,
        )
    return prod_idx + 1


def _pull_display_frame_for_pose(
    cap: JetsonGstNvmmCapture,
    store: PoseStore,
    target_seq: int,
    *,
    timeout_sec: float,
) -> tuple[Any, int, PoseDisplaySample] | None:
    deadline = time.monotonic() + max(0.01, float(timeout_sec))
    while time.monotonic() < deadline:
        ok, frame, capture_seq = cap.read_display_nv12()
        if not ok or frame is None:
            time.sleep(0.001)
            continue
        seq = int(capture_seq)
        if seq < target_seq:
            try:
                frame.release()
            except Exception:
                pass
            continue
        if seq == target_seq:
            pose = store.match(target_seq, policy="exact")
            if pose is None:
                try:
                    frame.release()
                except Exception:
                    pass
                return None
            return frame, seq, pose
        pose = store.match(seq, policy="exact")
        if pose is not None:
            return frame, seq, pose
        try:
            frame.release()
        except Exception:
            pass
    return None


def _display_use_cpu_bgr(cfg: Route2SessionConfig) -> bool:
    return bool(cfg.cpu_preview)


def _nv12_to_slot_tick(
    cfg: Route2SessionConfig,
    frame_slot: DisplayFrameSlot,
    frame: Any,
    capture_seq: int,
    *,
    pull_ms: float,
    wait_ms: float = 0.0,
    vic: DisplayVicEyeBgr | None = None,
) -> DisplayTick | None:
    """NV12 → slot 双缓冲（横幅单眼）。竖装转正在 live_jpeg 线程 CPU 做。"""
    h, w = display_preview_shape(frame, eye_rotate="none")
    write_buf = frame_slot.acquire_write_buffer((h, w, 3))
    if write_buf is None:
        return None
    if vic is not None:
        import time

        t0 = time.perf_counter()
        try:
            eye_bgr = vic.eye_bgr_from_sbs(frame, display_eye=cfg.display_eye)
        except Exception:
            return None
        preview_ms = (time.perf_counter() - t0) * 1000.0
        if eye_bgr.shape[:2] != (h, w):
            return None
        write_buf[:] = eye_bgr
    else:
        use_cpu = _display_use_cpu_bgr(cfg)
        try:
            _, preview_ms = nv12_display_frame_to_preview_bgr(
                frame,
                max_width=0,
                use_cpu=use_cpu,
                dst=write_buf,
                display_eye=cfg.display_eye,
                eye_rotate="none",
            )
        except Exception:
            return None
    return DisplayTick(
        capture_seq=int(capture_seq),
        bgr=write_buf,
        pose=None,
        pull_ms=float(pull_ms),
        preview_ms=float(preview_ms),
        wait_ms=float(wait_ms),
    )


def _nv12_to_slot_tick_gl(
    frame: Any,
    capture_seq: int,
    *,
    pull_ms: float,
    wait_ms: float = 0.0,
) -> DisplayTick | None:
    """P4：NV12 fd 入 slot，UI EGL+GL 直显（零 BGR 转换）。"""
    if frame.dmabuf_fd is None:
        return None
    return DisplayTick(
        capture_seq=int(capture_seq),
        nv12=frame,
        nv12_sbs=_is_sbs_full_frame(frame),
        pose=None,
        pull_ms=float(pull_ms),
        preview_ms=0.0,
        wait_ms=float(wait_ms),
    )
