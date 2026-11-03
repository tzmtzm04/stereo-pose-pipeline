"""Split display：出图 producer（daemon）+ UI 主线程 consumer。"""

from __future__ import annotations

import threading
import traceback
from typing import TextIO

from app.route2.session_config import Route2SessionConfig
from core.display.display_eye import display_eye_name
from core.display.display_tick import DisplayFrameSlot
from core.display.live_compose import OverlayFn
from core.display.pose_store import PoseStore
from core.display.video_ring import (
    DEFAULT_DISPLAY_VIDEO_RING_FRAMES,
    DEFAULT_DISPLAY_VIDEO_RING_FRAMES_NV12,
    DisplayVideoRing,
)
from domain.l3.runtime import L3Runtime
from orchestration.display_ui import set_embedded_preview_mode, teardown_display_window
from orchestration.worker_join import join_pipeline_workers
from orchestration.workers.display_producer_worker import run_display_producer_worker
from platform.jetson.capture import JetsonGstNvmmCapture
from ui.gl_preview_worker import run_ui_gl_preview_worker
from ui.preview_worker import run_ui_preview_worker


def _resolve_jpeg_slot():
    from app.sdk._registry import get_realtime_registry

    return get_realtime_registry().jpeg_slot()


def run_split_display_worker(
    cfg: Route2SessionConfig,
    cap: JetsonGstNvmmCapture,
    store: PoseStore,
    stop: threading.Event,
    quit_evt: threading.Event,
    metrics_fp: TextIO | None,
    *,
    l3: L3Runtime | None = None,
    overlay_fn: OverlayFn | None = None,
) -> None:
    """策略 D：prod 纯视频入环；无骨架原画直显，有骨架 pose→video exact 叠图。"""
    slot_depth = max(2, int(cfg.pipeline_queue_depth))
    frame_slot = DisplayFrameSlot(depth=slot_depth)

    gl_display = None
    use_gl = False
    if cfg.display_use_egl_gl:
        teardown_display_window()
        try:
            from core.display.hw_gl_display import create_hw_gl_display

            gl_display = create_hw_gl_display(cfg)
            use_gl = True
            print("ui EGL+GL 窗口就绪", flush=True)
            if cfg.embedded_preview:
                from app.sdk._bind import register_gl_display

                register_gl_display(gl_display)
                # create 已 override_redirect+unmapped；嵌套几何由 UI 线程 apply
                print(
                    "embedded preview: 黑盒 GL 已注册（屏外预热，就绪后再移入 Host 矩形）",
                    flush=True,
                )
        except Exception as exc:
            print(
                f"EGL+GL 窗口创建失败，回退 OpenCV imshow: {exc}",
                flush=True,
            )
            traceback.print_exc()
            gl_display = None

    if use_gl:
        ring_frames = DEFAULT_DISPLAY_VIDEO_RING_FRAMES_NV12
    elif cfg.cpu_jpeg_preview:
        # 嵌入 JPEG：ring 太浅时 pose 落后会被挤掉 → 叠不上/追赶空转
        ring_frames = 24
    elif cfg.embedded_preview:
        # 嵌入 OpenCV 窗贴洞：短环 + UI 跳最新 exact 对
        ring_frames = 8
    else:
        ring_frames = max(DEFAULT_DISPLAY_VIDEO_RING_FRAMES, slot_depth * 8)
    video_ring = DisplayVideoRing(max_frames=ring_frames, nv12_fd=use_gl)
    mode_tag = (
        "cpu_jpeg"
        if cfg.cpu_jpeg_preview
        else ("gl" if use_gl else ("embed" if cfg.embedded_preview else "opencv"))
    )
    print(
        f"split display 启动 eye={display_eye_name(cfg)} "
        f"mode={mode_tag}（prod daemon + ui ring={ring_frames}"
        f"{' nv12' if use_gl else ''}）",
        flush=True,
    )

    producer_th = threading.Thread(
        target=run_display_producer_worker,
        kwargs={
            "cfg": cfg,
            "cap": cap,
            "store": store,
            "frame_slot": frame_slot,
            "stop": stop,
            "wait_pose": False,
            "publish_nv12": use_gl,
            "video_ring": video_ring,
        },
        name="route2_display_producer",
        daemon=True,
    )
    producer_th.start()

    try:
        if use_gl:
            # 与 Route2 默认一致：黑盒可视化逻辑不变
            run_ui_gl_preview_worker(
                cfg,
                frame_slot,
                store,
                stop,
                quit_evt,
                metrics_fp,
                l3=l3,
                producer_th=producer_th,
                gl=gl_display,
                video_ring=video_ring,
            )
        elif cfg.cpu_jpeg_preview:
            teardown_display_window()
            from app.sdk.jpeg_slot import LatestJpegSlot
            from app.sdk.live_jpeg import run_cpu_jpeg_embed_ui_worker
            from app.sdk._bind import register_jpeg_slot

            jpeg_slot = _resolve_jpeg_slot()
            if jpeg_slot is None:
                jpeg_slot = LatestJpegSlot()
                register_jpeg_slot(jpeg_slot)
            print(
                "cpu_jpeg preview: 盒外 live_compose → /live.jpg（无 route2_pose3d 窗）",
                flush=True,
            )
            run_cpu_jpeg_embed_ui_worker(
                cfg,
                frame_slot,
                store,
                stop,
                quit_evt,
                video_ring=video_ring,
                jpeg_slot=jpeg_slot,
                metrics_fp=metrics_fp,
                producer_th=producer_th,
            )
        else:
            if cfg.embedded_preview:
                set_embedded_preview_mode(True)
                print(
                    "embedded preview: OpenCV CPU 窗（屏外起步，Host 贴采集洞；"
                    "BGR 环，不占 NV12/EGL）",
                    flush=True,
                )
            run_ui_preview_worker(
                cfg,
                frame_slot,
                store,
                stop,
                quit_evt,
                metrics_fp,
                l3=l3,
                overlay_fn=overlay_fn,
                producer_th=producer_th,
                video_ring=video_ring,
            )
    except Exception as exc:
        print(f"ui preview 异常退出: {exc}", flush=True)
        traceback.print_exc()
        quit_evt.set()
        stop.set()
    finally:
        stop.set()
        quit_evt.set()
        video_ring.close()
        frame_slot.close()
        join_pipeline_workers([("display_producer", producer_th)], stop=stop)
        if gl_display is not None:
            try:
                gl_display.close()
            except Exception:
                pass
