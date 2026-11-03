"""推理结果 → DisplayPacket + L1/L3 回调。"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from core.infer.pose3d_pipeline import StereoPose3DResult
from domain.l1.runtime import L1Runtime
from domain.l3.runtime import L3Runtime
from orchestration.types import DisplayPacket, FrameTiming


def build_display_packet(
    pose: StereoPose3DResult,
    *,
    l1: L1Runtime | None = None,
    l3: L3Runtime | None = None,
    preview_bgr: np.ndarray | None = None,
    sbs: Any | None = None,
    preview_keepalive: Any | None = None,
) -> DisplayPacket:
    t0 = time.perf_counter()
    timing = FrameTiming()
    timing.from_infer_result(pose)
    if l1 is not None:
        l1.on_pose3d_frame(
            capture_seq=pose.infer.capture_seq,
            infer_seq=pose.infer.frame_id,
            kpts3d=pose.kpts3d,
        )
        if l3 is not None:
            l3.on_l1_pose3d_frame(l1)
    consumer_ms = (time.perf_counter() - t0) * 1000.0
    packet = DisplayPacket(
        pose=pose,
        timing=timing,
        kpts3d=pose.kpts3d,
        preview_bgr=preview_bgr,
        sbs=sbs,
        _preview_keepalive=preview_keepalive,
    )
    packet.timing.consumer_ms = consumer_ms
    return packet
