"""应用 SDK：实时 / 离线黑盒薄封装（L-App 唯一入口）。

实时黑盒只出数据（SessionBundle + PoseStore）；可视化默认盒外 cpu_jpeg（/live.jpg）。
HTML 与其它编排代码请只 import 本包公开符号，勿直达 core/ / orchestration/。
"""

from __future__ import annotations

from app.sdk.offline import JobHandle, OfflineJob
from app.sdk.realtime import EmbeddedPreview, RealtimeSession, SessionHandle
from app.sdk.types import (
    FinalizeResult,
    JobStatus,
    MotionSegmentRef,
    OfflineArtifacts,
    PoseDrainStats,
    PoseFrameDTO,
    PreviewConfig,
    PreviewSyncResult,
    RealtimeStatus,
    ScreenRect,
    SegmentMark,
)

__all__ = [
    "EmbeddedPreview",
    "FinalizeResult",
    "JobHandle",
    "JobStatus",
    "MotionSegmentRef",
    "OfflineArtifacts",
    "OfflineJob",
    "PoseDrainStats",
    "PoseFrameDTO",
    "PreviewConfig",
    "PreviewSyncResult",
    "RealtimeSession",
    "RealtimeStatus",
    "ScreenRect",
    "SegmentMark",
    "SessionHandle",
]
