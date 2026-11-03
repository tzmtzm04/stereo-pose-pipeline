"""SDK 公开 DTO（与 contracts/ 互补，不含黑盒内部类型）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class ScreenRect:
    x: int
    y: int
    width: int
    height: int


@dataclass(frozen=True)
class PreviewConfig:
    """嵌套预览（Host 侧消费盒外可视化；黑盒本身只出 PoseStore + session 落盘）。"""

    inset_px: int = 0
    display_width: int = 720
    display_height: int = 1280
    present_min_renders: int = 8
    present_stable_polls: int = 3
    preview_mode: Literal["cpu_jpeg", "gl_embed"] = "cpu_jpeg"


@dataclass(frozen=True)
class RealtimeStatus:
    session_id: str
    running: bool
    latest_capture_seq: int
    pose_frames: int
    tri_ok_rate: float
    preview_ready: bool
    error: str | None = None


@dataclass(frozen=True)
class PoseDrainStats:
    samples: int
    tri_ok: int
    last_capture_seq: int


@dataclass(frozen=True)
class PoseFrameDTO:
    """流式 3D pose 单帧（Host 边采边分析用）。"""

    capture_seq: int
    infer_seq: int
    tri_ok: bool
    kpts3d: tuple[tuple[float, float, float, float], ...] | None = None


@dataclass(frozen=True)
class SegmentMark:
    label: str
    event: Literal["start", "end"]
    capture_seq: int


@dataclass(frozen=True)
class PreviewSyncResult:
    presented: bool
    warming: bool
    geom: tuple[int, int, int, int] | None = None


JobState = Literal["pending", "running", "done", "failed"]


@dataclass(frozen=True)
class JobStatus:
    state: JobState
    stage: int | None = None
    error: str | None = None


@dataclass(frozen=True)
class OfflineArtifacts:
    session_dir: Path
    pose2d: Path | None = None
    clean_2d: Path | None = None
    tri_filtered_3d: Path | None = None
    constrained_3d: Path | None = None  # 兼容别名，等同 tri_filtered_3d
    summaries: dict[str, Path] = field(default_factory=dict)


@dataclass(frozen=True)
class MotionSegmentRef:
    """切分后的单条运动段（可作离线 SessionBundle 根目录）。"""

    index: int
    label: str
    start_capture_seq: int
    end_capture_seq: int
    stem: str
    clip_dir: Path


@dataclass(frozen=True)
class FinalizeResult:
    """实时结束后切分结果。"""

    session_id: str
    session_dir: Path
    segments: list = field(default_factory=list)
    motion_segments: list[MotionSegmentRef] = field(default_factory=list)
    clips_manifest: Path | None = None
    motion_segments_json: Path | None = None
