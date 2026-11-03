"""DisplayPacket 生命周期（SBS fd / GPU BGR 释放）。"""

from __future__ import annotations

from typing import Any

import cv2

from orchestration.display_ui import resolve_display_size
from orchestration.types import DisplayPacket


def release_packet_preview(packet: DisplayPacket | None) -> None:
    if packet is None:
        return
    if packet._preview_keepalive is not None:
        from platform.jetson.nv12_frame import _release_keepalive

        _release_keepalive(packet._preview_keepalive)
        packet._preview_keepalive = None
    packet.preview_bgr = None


def release_packet_sbs(packet: DisplayPacket | None) -> None:
    if packet is None:
        return
    if packet.sbs is not None:
        try:
            packet.sbs.release()
        except Exception:
            pass
        packet.sbs = None
    release_packet_preview(packet)


def preview_from_packet(cfg, packet: DisplayPacket) -> tuple[Any, float]:
    """display 线程：GPU BGR 已与窗口同尺寸时零 resize。"""
    import time

    dw, dh = resolve_display_size(cfg)

    if packet.preview_bgr is not None:
        bgr = packet.preview_bgr
        h, w = bgr.shape[:2]
        if w == dw and h == dh:
            return bgr, float(packet.timing.preview_ms)
        t0 = time.perf_counter()
        out = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
        resize_ms = (time.perf_counter() - t0) * 1000.0
        return out, float(packet.timing.preview_ms) + resize_ms

    from core.display.sbs_preview import sbs_preview_bgr

    if packet.sbs is None:
        return None, 0.0
    t0 = time.perf_counter()
    bgr = sbs_preview_bgr(packet.sbs, max_width=dw if dw > 0 else 0)
    preview_ms = (time.perf_counter() - t0) * 1000.0
    if bgr is None:
        return None, preview_ms
    h, w = bgr.shape[:2]
    if w == dw and h == dh:
        return bgr, preview_ms
    t1 = time.perf_counter()
    out = cv2.resize(bgr, (dw, dh), interpolation=cv2.INTER_AREA)
    preview_ms += (time.perf_counter() - t1) * 1000.0
    return out, preview_ms
