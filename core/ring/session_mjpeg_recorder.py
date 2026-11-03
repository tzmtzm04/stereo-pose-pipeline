"""MJPEG session 录像：ZED grab ingress submit。"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from platform.jetson.nv12_frame import GstBufferKeepalive

from contracts.blackbox import BLACKBOX_CONTRACT_VERSION
from contracts.session_bundle import (
    RING_MODE_SESSION_MJPEG,
    SESSION_MJPEG_FILENAME,
    SessionBundle,
    SessionMetaFields,
)
from core.ring.encoded_ring_buffer import EncodedPacket, EncodedRingBuffer
from core.ring.session_disk_writer import (
    SessionDiskWriter,
    SessionEncodedRingBuffer,
    finalize_session_meta,
    write_session_meta,
)
from domain.l1.pose_history import Pose3DSample
from domain.l1.pose_session_log import L1PoseSessionLog
from domain.l1.session_recorder import SessionStats

logger = logging.getLogger(__name__)

DEFAULT_MJPEG_QUEUE_DEPTH = 64
DEFAULT_SESSION_MJPEG_PACKETS = 64
INGRESS_THREAD_NICE = 10


class MjpegIngressWorker:
    """ZED 对齐：probe 只 ref buffer 入队，后台线程拷贝 JPEG → submit。"""

    def __init__(
        self,
        submit_fn: Callable[[int, bytes], bool],
        buffer_payload_fn: Callable[[Any], bytes | None],
        *,
        queue_depth: int = DEFAULT_MJPEG_QUEUE_DEPTH,
    ) -> None:
        self._submit_fn = submit_fn
        self._buffer_payload_fn = buffer_payload_fn
        depth = max(8, int(queue_depth))
        self._q: queue.Queue[
            tuple[int, GstBufferKeepalive] | None
        ] = queue.Queue(maxsize=depth)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._enqueued = 0
        self._submitted = 0
        self._queue_dropped = 0
        self._payload_fail = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="route2_mjpeg_ingress",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._drain_queue()

    def enqueue(self, capture_seq: int, buf: Any) -> None:
        """GStreamer probe 调用：非阻塞 ref buffer 入队。"""
        if self._stop.is_set() or buf is None:
            return
        item = (int(capture_seq), GstBufferKeepalive(buf))
        try:
            self._q.put_nowait(item)
            self._enqueued += 1
            return
        except queue.Full:
            pass
        try:
            old = self._q.get_nowait()
            if old is not None:
                old[1].unref()
            self._queue_dropped += 1
        except queue.Empty:
            pass
        try:
            self._q.put_nowait(item)
            self._enqueued += 1
        except queue.Full:
            item[1].unref()
            self._queue_dropped += 1

    def stats(self) -> dict:
        return {
            "mjpeg_ingress_enqueued": self._enqueued,
            "mjpeg_ingress_submitted": self._submitted,
            "mjpeg_ingress_q_dropped": self._queue_dropped,
            "mjpeg_ingress_payload_fail": self._payload_fail,
            "mjpeg_ingress_q": self._q.qsize(),
        }

    def _drain_queue(self) -> None:
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is None:
                continue
            item[1].unref()

    def _run(self) -> None:
        import os

        try:
            os.nice(INGRESS_THREAD_NICE)
        except OSError:
            pass
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                break
            capture_seq, keepalive = item
            try:
                buf = keepalive.buf
                if buf is None:
                    self._payload_fail += 1
                    continue
                data = self._buffer_payload_fn(buf)
                if not data:
                    self._payload_fail += 1
                    continue
                if self._submit_fn(capture_seq, data):
                    self._submitted += 1
            except Exception as exc:
                self._payload_fail += 1
                logger.warning("mjpeg ingress worker: %s", exc)
            finally:
                keepalive.unref()


@dataclass
class SessionMjpegRecordSession:
    """MJPEG session 录像（grab ingress）。"""

    ring: EncodedRingBuffer
    disk_writer: SessionDiskWriter
    bundle: SessionBundle
    fps_nominal: int
    width: int
    height: int
    pose_log: L1PoseSessionLog
    ingress_worker: MjpegIngressWorker | None = None
    started_ts: float = field(default_factory=time.time)
    _packets_submitted: int = field(default=0, init=False)
    _ingress_min_seq: int | None = field(default=None, init=False)
    _ingress_max_seq: int | None = field(default=None, init=False)
    _first_ingress_ts: float | None = field(default=None, init=False)
    _last_ingress_ts: float | None = field(default=None, init=False)

    def note_ingress_frame(self, capture_seq: int) -> None:
        """记录 ingress 序号与时间窗（用于 duration_sec ≈ 稳态墙钟 fps）。"""
        seq = int(capture_seq)
        now = time.time()
        if self._ingress_min_seq is None:
            self._ingress_min_seq = seq
            self._first_ingress_ts = now
        self._ingress_max_seq = seq
        self._last_ingress_ts = now

    def recording_duration_sec(self) -> float:
        """首帧→末帧 capture_seq 跨度 / fps（与相机时间轴一致）。"""
        fps = max(1, int(self.fps_nominal))
        if self._ingress_min_seq is not None and self._ingress_max_seq is not None:
            span = int(self._ingress_max_seq) - int(self._ingress_min_seq) + 1
            if span > 0:
                return span / float(fps)
        if self._first_ingress_ts is not None and self._last_ingress_ts is not None:
            dt = float(self._last_ingress_ts) - float(self._first_ingress_ts)
            if dt > 0:
                return dt
        return max(1.0 / float(fps), time.time() - float(self.started_ts))

    def ingest_pose(self, sample: Pose3DSample) -> None:
        self.pose_log.append_sample(sample)

    def submit_ingress_packet(self, capture_seq: int, data: bytes) -> bool:
        """mjpeg_tee ingress 非阻塞 submit（写盘由 disk writer 异步）。"""
        if not data:
            return False
        self.note_ingress_frame(capture_seq)
        pkt = EncodedPacket(
            capture_seq=int(capture_seq),
            pts_ns=0,
            is_keyframe=True,
            data=bytes(data),
        )
        self.ring.push(pkt)
        self._packets_submitted += 1
        return True

    def stats(self) -> dict:
        st = self.ring.stats()
        st["fps_nominal"] = self.fps_nominal
        st["width"] = self.width
        st["height"] = self.height
        frames = int(self._packets_submitted)
        st["mjpeg_packets_pulled"] = frames
        st["mjpeg_pull_fail_streak"] = 0
        if self.ingress_worker is not None:
            st.update(self.ingress_worker.stats())
        st["frames_submitted"] = frames
        st.update(self.disk_writer.stats())
        return st

    def stop(self) -> SessionStats:
        if self.ingress_worker is not None:
            self.ingress_worker.stop()
            self.ingress_worker = None
        self.disk_writer.stop()
        st = self.stats()
        return SessionStats(
            frames_submitted=int(st.get("frames_submitted", 0)),
            enc_dropped=int(st.get("session_writer_dropped", 0)),
            h264_bytes=int(st.get("session_h264_bytes", 0)),
            h264_packets=int(st.get("session_h264_packets", 0)),
        )


def create_session_mjpeg_record(
    *,
    session_dir: Path,
    width: int,
    height: int,
    fps: int = 60,
    device: str = "/dev/video0",
    calib_path: Path,
    session_id: Optional[str] = None,
    mjpeg_queue_depth: int = DEFAULT_MJPEG_QUEUE_DEPTH,
    session_mjpeg_packets: int = DEFAULT_SESSION_MJPEG_PACKETS,
) -> SessionMjpegRecordSession:
    """启动 session.mjpeg（ZED grab ingress）。"""
    bundle = SessionBundle.from_dir(session_dir)
    bundle.session_dir.mkdir(parents=True, exist_ok=True)

    for path in (bundle.mjpeg_path, bundle.index_path):
        if path.is_file():
            path.unlink()

    disk_writer = SessionDiskWriter(bundle.mjpeg_path, bundle.index_path)
    disk_writer.start()

    fps_i = max(1, int(fps))
    ring_cap = max(16, int(session_mjpeg_packets))

    ring = SessionEncodedRingBuffer(
        disk_writer=disk_writer,
        max_packets=ring_cap,
        iframe_interval=1,
    )

    write_session_meta(
        bundle.meta_path,
        SessionMetaFields(
            width=int(width),
            height=int(height),
            fps=fps_i,
            bitrate=0,
            iframe_interval=1,
            device=device,
            calib_path=calib_path,
            session_id=session_id,
            ring_mode=RING_MODE_SESSION_MJPEG,
            enc_mode="mjpeg_grab_ingress",
            layout="mjpeg_native",
            enc_queue_depth=max(8, int(mjpeg_queue_depth)),
            enc_nvbuf_pool_size=0,
            video_file=SESSION_MJPEG_FILENAME,
            blackbox_contract_version=BLACKBOX_CONTRACT_VERSION,
        ),
    )

    logger.info(
        "Session MJPEG record (mjpeg_grab_ingress): %s %dx%d fps=%d ring=%d",
        bundle.session_dir,
        width,
        height,
        fps_i,
        ring_cap,
    )

    return SessionMjpegRecordSession(
        ring=ring,
        disk_writer=disk_writer,
        bundle=bundle,
        fps_nominal=fps_i,
        width=int(width),
        height=int(height),
        pose_log=L1PoseSessionLog(bundle.pose_path),
    )


__all__ = [
    "MjpegIngressWorker",
    "SessionMjpegRecordSession",
    "create_session_mjpeg_record",
    "finalize_session_meta",
]
