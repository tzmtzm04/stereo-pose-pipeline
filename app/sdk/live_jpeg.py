"""页内实况：DisplayVideoRing（NV12/BGR）→ JPEG，供 Host 轮询（SDK 参考预览）。

叠骨架：pose→video（同 capture_seq exact），不用 latest_le。
infer 落后导致 pose 已滑出 ring 时：追赶到环内 pose；仍没有则出原画，
保证固定帧率发布，避免画面卡死。
"""

from __future__ import annotations

import threading
import time
from typing import TextIO

import cv2
import numpy as np

from app.sdk.jpeg_slot import LatestJpegSlot
from app.route2.session_config import Route2SessionConfig
from core.display.display_tick import DisplayFrameSlot, DisplayTick
from core.display.live_compose import (
    compose_display_frame,
    resolve_pose_sample_for_display,
)
from core.display.pose_store import PoseDisplaySample, PoseStore
from core.display.sbs_preview import nv12_display_frame_to_preview_bgr
from core.display.video_ring import DisplayVideoRing, release_display_tick
from core.geometry.eye_mount_rotate import apply_eye_mount_rotate, parse_eye_mount_rotate

__all__ = [
    "LatestJpegSlot",
    "run_cpu_jpeg_embed_ui_worker",
]


def run_cpu_jpeg_embed_ui_worker(
    cfg: Route2SessionConfig,
    frame_slot: DisplayFrameSlot,
    store: PoseStore,
    stop: threading.Event,
    quit_evt: threading.Event,
    *,
    video_ring: DisplayVideoRing,
    jpeg_slot: LatestJpegSlot,
    metrics_fp: TextIO | None = None,
    producer_th: threading.Thread | None = None,
    jpeg_quality: int = 75,
    max_fps: float = 12.0,
) -> None:
    """固定帧率发布 JPEG；有同 seq 视频则叠骨架，否则原画。"""
    _ = (frame_slot, metrics_fp)
    print(
        f"cpu_jpeg embed UI：页内 JPEG 轮询 "
        f"max_fps={max_fps} q={jpeg_quality} nv12={video_ring.nv12_fd} "
        f"pose→video(+原画回退)",
        flush=True,
    )
    min_dt = 1.0 / max(1.0, float(max_fps))
    last_pub = 0.0
    last_seq = -1
    published = 0
    overlay_hits = 0
    eye = str(getattr(cfg, "display_eye", "left") or "left")
    ring_depth = max(4, int(video_ring.max_frames))

    deadline = time.monotonic() + 30.0
    while not stop.is_set() and not quit_evt.is_set():
        if video_ring.latest_capture_seq >= 0:
            break
        if time.monotonic() > deadline:
            print("cpu_jpeg embed：等待首帧超时", flush=True)
            break
        time.sleep(0.02)

    print(
        f"cpu_jpeg embed bootstrap live={video_ring.latest_capture_seq} "
        f"pose={store.latest_capture_seq} ring={ring_depth}",
        flush=True,
    )

    while not quit_evt.is_set() and not stop.is_set():
        if producer_th is not None and not producer_th.is_alive():
            print("cpu_jpeg embed UI：display producer 已停止", flush=True)
            stop.set()
            break

        now = time.perf_counter()
        sleep_for = min_dt - (now - last_pub)
        if sleep_for > 0.001:
            time.sleep(sleep_for)

        live = int(video_ring.latest_capture_seq)
        if live < 0:
            time.sleep(0.01)
            continue

        oldest = max(0, live - ring_depth + 1)
        tick: DisplayTick | None = None
        tick_owned = False
        overlay_pose = None
        try:
            pose = _pick_pose_in_ring(store, last_seq=last_seq, oldest=oldest, live=live)
            if pose is not None:
                target = int(pose.capture_seq)
                taken = video_ring.take(target)
                if taken is not None:
                    tick = taken
                    tick_owned = True
                    resolved = resolve_pose_sample_for_display(cfg, pose)
                    overlay_pose = resolved.pose
                    if overlay_pose is not None:
                        overlay_hits += 1
                else:
                    # 该 pose 已滑出：推进游标，本帧走原画
                    last_seq = max(last_seq, target)

            if tick is None:
                tick = video_ring.take_newest_after(last_seq)
                if tick is not None:
                    tick_owned = True
                else:
                    tick = video_ring.get(live)
                    tick_owned = False
                if tick is None:
                    time.sleep(0.005)
                    continue
                # 原画回退：若恰好有同 seq pose，仍可叠
                if overlay_pose is None:
                    exact = store.match(int(tick.capture_seq), policy="exact")
                    if exact is not None:
                        resolved = resolve_pose_sample_for_display(cfg, exact)
                        overlay_pose = resolved.pose
                        if overlay_pose is not None:
                            overlay_hits += 1

            bgr = _tick_to_bgr(cfg, tick, eye=eye)
            if bgr is None:
                continue
            rot = parse_eye_mount_rotate(
                str(getattr(cfg, "eye_rotate", "none") or "none")
            )
            # infer VIC 已输出竖图关键点；预览先 CPU 转正再叠，坐标一致
            if rot != "none":
                bgr = apply_eye_mount_rotate(bgr, rot)
            composite = compose_display_frame(
                cfg, bgr, overlay_pose, fill="cover"
            )
            ok, buf = cv2.imencode(
                ".jpg",
                composite,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            if ok:
                jpeg_slot.publish(buf.tobytes())
                published += 1
                last_pub = time.perf_counter()
                last_seq = max(last_seq, int(tick.capture_seq))
                if published == 1 or published % 60 == 0:
                    print(
                        f"cpu_jpeg embed jpeg n={published} seq={last_seq} "
                        f"live={live} pose={store.latest_capture_seq} "
                        f"overlay_hits={overlay_hits}",
                        flush=True,
                    )
        finally:
            if tick_owned and tick is not None:
                release_display_tick(tick)

    print(
        f"cpu_jpeg embed UI 结束 published={published} overlay_hits={overlay_hits}",
        flush=True,
    )


def _pick_pose_in_ring(
    store: PoseStore,
    *,
    last_seq: int,
    oldest: int,
    live: int,
) -> PoseDisplaySample | None:
    """取 ring 内、且 > last_seq 的最新可用 pose（非阻塞，可追赶陈旧 pose）。"""
    _ = live
    cursor = int(last_seq)
    # 快速跳过已滑出 ring 的 pose
    while True:
        nxt = store.next_pose_after(cursor)
        if nxt is None:
            break
        seq = int(nxt.capture_seq)
        if seq < int(oldest):
            cursor = seq
            continue
        # 取 ring 内最新 pose（追到队尾）
        best = nxt
        while True:
            more = store.next_pose_after(int(best.capture_seq))
            if more is None:
                break
            if int(more.capture_seq) > int(live):
                break
            if int(more.capture_seq) < int(oldest):
                best = more
                continue
            best = more
        return best
    # 没有更新的：若 latest 仍在环内且尚未显示，用它
    latest = int(store.latest_capture_seq)
    if latest > int(last_seq) and latest >= int(oldest):
        return store.match(latest, policy="exact")
    return None


def _tick_to_bgr(
    cfg: Route2SessionConfig,
    tick: DisplayTick,
    *,
    eye: str,
) -> np.ndarray | None:
    if tick.bgr is not None:
        return tick.bgr
    if tick.nv12 is None:
        return None
    bgr, _ms = nv12_display_frame_to_preview_bgr(
        tick.nv12,
        max_width=int(cfg.display_width) if cfg.display_width > 0 else 640,
        use_cpu=bool(getattr(cfg, "cpu_preview", False)),
        display_eye=eye,
        eye_rotate=str(getattr(cfg, "eye_rotate", "none") or "none"),
    )
    return bgr
