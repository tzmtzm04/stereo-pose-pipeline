"""orchestration 绑定入口（避免 realtime ↔ pipeline 循环 import）。"""

from __future__ import annotations

import threading

from app.route2.session_config import Route2SessionConfig
from app.sdk._registry import get_realtime_registry


def bind_embedded_realtime(
    cfg: Route2SessionConfig,
    pose_store,
    stop: threading.Event,
    quit_evt: threading.Event,
) -> None:
    if pose_store is None:
        return
    reg = get_realtime_registry()
    reg.bind_pose_store(pose_store)
    if cfg.embedded_preview or cfg.cpu_jpeg_preview:
        reg.bind_pipeline_control(stop, quit_evt)


def register_gl_display(gl) -> None:
    get_realtime_registry().bind_gl_display(gl)


def register_jpeg_slot(slot) -> None:
    get_realtime_registry().bind_jpeg_slot(slot)
