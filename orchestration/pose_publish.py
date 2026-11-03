"""infer 完成后写入 PoseStore 并触发 L1/L3（与 display 解耦）。"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

from core.display.pose_store import PoseDisplaySample, PoseStore
from core.infer.pose3d_pipeline import StereoPose3DResult
from domain.l1.runtime import L1Runtime
from domain.l3.runtime import L3Runtime

if TYPE_CHECKING:
    from orchestration.c2_auto_stop import C2AutoStopGate

_pose_audit_count = 0


def _maybe_pose_seq_audit(capture_seq: int, infer_seq: int) -> None:
    global _pose_audit_count
    if os.environ.get("ROUTE2_SEQ_AUDIT", "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return
    _pose_audit_count += 1
    every = 60
    raw = os.environ.get("ROUTE2_SEQ_AUDIT_EVERY", "60").strip()
    try:
        every = max(1, int(raw))
    except ValueError:
        pass
    if _pose_audit_count != 1 and _pose_audit_count % every != 0:
        return
    print(
        f"seq_audit pose_publish capture_seq={int(capture_seq)} "
        f"infer_seq={int(infer_seq)} n={_pose_audit_count}",
        flush=True,
    )


def publish_infer_pose(
    pose: StereoPose3DResult,
    store: PoseStore,
    *,
    l1: L1Runtime | None = None,
    l3: L3Runtime | None = None,
    c2_auto_stop: C2AutoStopGate | None = None,
) -> PoseDisplaySample:
    t0 = time.perf_counter()
    sample = store.put_from_pose(pose)
    _maybe_pose_seq_audit(pose.infer.capture_seq, pose.infer.frame_id)
    if l1 is not None:
        l1.on_pose3d_frame(
            capture_seq=pose.infer.capture_seq,
            infer_seq=pose.infer.frame_id,
            kpts3d=pose.kpts3d,
            tri_ok=bool(pose.tri_ok),
            det_left=pose.det_left,
            det_right=pose.det_right,
        )
        if l3 is not None:
            l3.on_l1_pose3d_frame(l1)
    if c2_auto_stop is not None:
        c2_auto_stop.on_pose(sample)
    _ = (time.perf_counter() - t0) * 1000.0
    return sample
