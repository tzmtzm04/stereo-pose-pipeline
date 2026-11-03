"""Session 录制抽象（对标 ZED enableRecording / ingestDataIntoSVO）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol, TYPE_CHECKING

from contracts.session_bundle import SessionBundle
from domain.l1.pose_history import Pose3DSample
from platform.jetson.nv12_frame import (
    GstSampleKeepalive,
    Nv12SbsFrame,
    gst_sample_keepalive_from_any,
)

if TYPE_CHECKING:
    from platform.jetson.session_enc_nvbuf_pool import EncPoolLease, SessionEncNvBufPool


@dataclass(frozen=True)
class RecordingFrame:
    """grab 路径：enc NvBuf 池 / 零拷贝 ref / 独立 buffer（copy 回退）。"""

    capture_seq: int
    keepalive: GstSampleKeepalive | None = None
    buffer: Any | None = None
    pts_ns: int = 0
    pool_lease: "EncPoolLease | None" = None
    deferred_pool_blit: bool = False
    deferred_gst_copy: bool = False


@dataclass(frozen=True)
class SessionStats:
    frames_submitted: int
    enc_dropped: int
    h264_bytes: int
    h264_packets: int


class SessionRecorder(Protocol):
    def start(self, bundle: SessionBundle) -> None: ...

    def submit_frame(self, frame: RecordingFrame) -> None: ...

    def ingest_pose(self, sample: Pose3DSample) -> None: ...

    def stop(self) -> SessionStats: ...

    def stats(self) -> dict: ...


def _keepalive_from_sbs(sbs: Nv12SbsFrame) -> GstSampleKeepalive | None:
    ka = gst_sample_keepalive_from_any(sbs._keepalive)
    if ka is not None:
        return ka
    if sbs.full is not None:
        return gst_sample_keepalive_from_any(sbs.full._keepalive)
    return None


def _pts_ns_from_sample(sample: Any) -> int:
    buf = sample.get_buffer()
    if buf is None:
        return 0
    try:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        if buf.pts != Gst.CLOCK_TIME_NONE:
            return int(buf.pts)
    except Exception:
        pass
    return 0


def recording_frame_from_sbs(
    sbs: Nv12SbsFrame,
    *,
    copy: bool = False,
    enc_pool: "SessionEncNvBufPool | None" = None,
    pool_deferred: bool = False,
    defer_copy: bool = False,
) -> RecordingFrame | None:
    """提交 NVENC：copy | defer copy | 同步 pool blit | deferred pool | ref。"""
    ka = _keepalive_from_sbs(sbs)
    if ka is None or ka.sample is None:
        return None
    sample = ka.sample
    buf = sample.get_buffer()
    if buf is None:
        return None
    pts_ns = _pts_ns_from_sample(sample)
    seq = int(sbs.capture_seq)
    if seq < 0:
        offset = int(buf.offset)
        if offset > 0:
            seq = offset - 1
    if pool_deferred or defer_copy:
        return RecordingFrame(
            capture_seq=seq,
            keepalive=ka.ref(),
            pts_ns=pts_ns,
            deferred_pool_blit=bool(pool_deferred),
            deferred_gst_copy=bool(defer_copy),
        )
    if copy:
        copied = buf.copy()
        if copied is None:
            return None
        return RecordingFrame(
            capture_seq=seq,
            buffer=copied,
            pts_ns=pts_ns,
        )
    if enc_pool is not None:
        lease = enc_pool.try_blit(sbs)
        if lease is None:
            return None
        return RecordingFrame(
            capture_seq=seq,
            buffer=lease.buffer,
            pts_ns=pts_ns,
            pool_lease=lease,
        )
    return RecordingFrame(
        capture_seq=seq,
        keepalive=ka.ref(),
        pts_ns=pts_ns,
    )
