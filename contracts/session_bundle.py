"""一次 live 运行的 Session Bundle 布局（对标 ZED SVO 目录）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from contracts.blackbox import BLACKBOX_CONTRACT_VERSION

SESSION_H264_FILENAME = "session.h264"
SESSION_MJPEG_FILENAME = "session.mjpeg"
SESSION_INDEX_FILENAME = "session_index.jsonl"
SESSION_POSE_FILENAME = "session_pose.jsonl"
SESSION_META_FILENAME = "session_meta.json"
CLIPS_MANIFEST_FILENAME = "clips_manifest.jsonl"

RING_MODE_SESSION_H264_GRAB = "session_h264_grab"
RING_MODE_SESSION_H264_TEE = "session_h264_tee"
RING_MODE_SESSION_MJPEG = "session_mjpeg"
RING_MODE_RAW_NV12_MEMORY = "raw_nv12_memory"


@dataclass(frozen=True)
class SessionBundle:
    """outputs/route2_clips/{ts}/ 下的标准产物集合。"""

    session_dir: Path

    @property
    def h264_path(self) -> Path:
        return self.session_dir / SESSION_H264_FILENAME

    @property
    def mjpeg_path(self) -> Path:
        return self.session_dir / SESSION_MJPEG_FILENAME

    @property
    def index_path(self) -> Path:
        return self.session_dir / SESSION_INDEX_FILENAME

    @property
    def pose_path(self) -> Path:
        return self.session_dir / SESSION_POSE_FILENAME

    @property
    def meta_path(self) -> Path:
        return self.session_dir / SESSION_META_FILENAME

    @property
    def manifest_path(self) -> Path:
        return self.session_dir / CLIPS_MANIFEST_FILENAME

    @classmethod
    def from_dir(cls, session_dir: Path) -> "SessionBundle":
        return cls(session_dir=Path(session_dir).expanduser().resolve())


@dataclass(frozen=True)
class SessionMetaFields:
    """session_meta.json 核心字段（写盘时填充）。"""

    width: int
    height: int
    fps: int
    bitrate: int
    iframe_interval: int
    device: str
    calib_path: Path
    ring_mode: str = RING_MODE_SESSION_H264_GRAB
    enc_mode: str = "grab_deferred"
    layout: Optional[str] = None
    session_id: Optional[str] = None
    enc_queue_depth: int = 4
    enc_nvbuf_pool_size: int = 0
    video_file: str = SESSION_H264_FILENAME
    index_file: str = SESSION_INDEX_FILENAME
    pose_file: str = SESSION_POSE_FILENAME
    blackbox_contract_version: int = BLACKBOX_CONTRACT_VERSION
    realtime_preview_mode: str | None = None
