"""Session 录像异步写盘：session.mjpeg + session_index.jsonl。"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

from contracts.session_bundle import SessionMetaFields
from core.ring.encoded_ring_buffer import EncodedPacket, EncodedRingBuffer

logger = logging.getLogger(__name__)

ENC_THREAD_NICE = 10
DISK_FLUSH_EVERY_PACKETS = 30


class SessionDiskWriter:
    """异步 append 编码包 + session_index.jsonl。"""

    def __init__(
        self,
        media_path: Path,
        index_path: Path,
        *,
        queue_depth: int = 64,
    ) -> None:
        self._media_path = Path(media_path)
        self._index_path = Path(index_path)
        self._media_path.parent.mkdir(parents=True, exist_ok=True)
        self._q: queue.Queue[EncodedPacket | None] = queue.Queue(
            maxsize=max(8, int(queue_depth))
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._offset = 0
        self._packets_written = 0
        self._bytes_written = 0
        self._dropped = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="route2_session_disk_writer",
            daemon=True,
        )
        self._thread.start()

    def submit(self, packet: EncodedPacket) -> None:
        if self._stop.is_set():
            return
        try:
            self._q.put_nowait(packet)
        except queue.Full:
            try:
                _ = self._q.get_nowait()
                self._dropped += 1
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(packet)
            except queue.Full:
                self._dropped += 1

    def stop(self) -> None:
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    def stats(self) -> dict:
        return {
            "session_h264_bytes": self._bytes_written,
            "session_h264_packets": self._packets_written,
            "session_writer_q": self._q.qsize(),
            "session_writer_dropped": self._dropped,
        }

    def _run(self) -> None:
        import os

        try:
            os.nice(ENC_THREAD_NICE)
        except OSError:
            pass
        pending_flush = 0
        with self._media_path.open("ab") as media_fp, self._index_path.open(
            "a", encoding="utf-8"
        ) as index_fp:
            while not self._stop.is_set():
                try:
                    pkt = self._q.get(timeout=0.2)
                except queue.Empty:
                    continue
                if pkt is None:
                    break
                offset = self._offset
                size = len(pkt.data)
                media_fp.write(pkt.data)
                index_fp.write(
                    json.dumps(
                        {
                            "offset": offset,
                            "size": size,
                            "capture_seq": pkt.capture_seq,
                            "pts_ns": pkt.pts_ns,
                            "is_keyframe": pkt.is_keyframe,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                self._offset += size
                self._packets_written += 1
                self._bytes_written += size
                pending_flush += 1
                if pending_flush >= DISK_FLUSH_EVERY_PACKETS:
                    media_fp.flush()
                    index_fp.flush()
                    pending_flush = 0
            media_fp.flush()
            index_fp.flush()


class SessionEncodedRingBuffer(EncodedRingBuffer):
    """小内存编码包环（写盘由 disk writer 负责）。"""

    def __init__(
        self,
        *,
        disk_writer: SessionDiskWriter,
        max_packets: int,
        iframe_interval: int,
    ) -> None:
        super().__init__(
            max_packets=max_packets,
            iframe_interval=iframe_interval,
        )
        self._disk_writer = disk_writer

    def push(self, packet: EncodedPacket) -> None:
        super().push(packet)
        self._disk_writer.submit(packet)


def write_session_meta(path: Path, meta: SessionMetaFields) -> None:
    body = {
        "width": int(meta.width),
        "height": int(meta.height),
        "fps": int(meta.fps),
        "bitrate": int(meta.bitrate),
        "iframe_interval": int(meta.iframe_interval),
        "enc_mode": str(meta.enc_mode),
        "enc_queue_depth": int(meta.enc_queue_depth),
        "enc_nvbuf_pool_size": int(meta.enc_nvbuf_pool_size),
        "device": str(meta.device),
        "calib_path": str(meta.calib_path),
        "session_id": meta.session_id,
        "ring_mode": str(meta.ring_mode),
        "created_ts": time.time(),
        "video_file": meta.video_file,
        "index_file": meta.index_file,
        "pose_file": meta.pose_file,
    }
    if meta.layout is not None:
        body["layout"] = str(meta.layout)
    body["blackbox_contract_version"] = int(meta.blackbox_contract_version)
    if meta.realtime_preview_mode is not None:
        body["realtime_preview_mode"] = str(meta.realtime_preview_mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize_session_meta(
    path: Path,
    *,
    frames_submitted: int,
    frames_recorded: int,
    frames_pose: int,
    duration_sec: float,
    enc_dropped: int = 0,
    h264_bytes: int = 0,
    duration_wall_sec: float | None = None,
) -> None:
    """shutdown 时合并 stop 统计到 session_meta.json。"""
    meta_path = Path(path)
    body: dict = {}
    if meta_path.is_file():
        try:
            body = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            body = {}
    body["stopped_ts"] = time.time()
    body["duration_sec"] = round(float(duration_sec), 3)
    if duration_wall_sec is not None:
        body["duration_wall_sec"] = round(float(duration_wall_sec), 3)
    if duration_sec > 0 and frames_submitted > 0:
        body["rec_fps"] = round(int(frames_submitted) / float(duration_sec), 2)
    body["frames_submitted"] = int(frames_submitted)
    body["frames_recorded"] = int(frames_recorded)
    body["frames_pose"] = int(frames_pose)
    body["enc_dropped"] = int(enc_dropped)
    body["enc_bytes"] = int(h264_bytes)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "SessionDiskWriter",
    "SessionEncodedRingBuffer",
    "write_session_meta",
    "finalize_session_meta",
]
