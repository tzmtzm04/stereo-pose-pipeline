"""L1 主管线附属运行时（形态 B：MJPEG session + pose log）。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from core.ring.encoded_ring_buffer import EncodedRingBuffer
from core.ring.session_disk_writer import finalize_session_meta
from core.ring.session_mjpeg_recorder import (
    SessionMjpegRecordSession,
    create_session_mjpeg_record,
)
from domain.l1.pose_history import Pose3DSample
from domain.l1.pose_session_log import L1PoseSessionLog
from domain.l3.params import L3Params

logger = logging.getLogger(__name__)


def _copy_det(det: dict | None) -> dict | None:
    if det is None:
        return None
    out = dict(det)
    kp = det.get("keypoints")
    if kp is not None:
        out["keypoints"] = np.asarray(kp, dtype=np.float32).copy()
    kp_hypo = det.get("keypoints_hypo")
    if kp_hypo is not None:
        out["keypoints_hypo"] = np.asarray(kp_hypo, dtype=np.float32).copy()
    return out


@dataclass(frozen=True)
class SessionMeta:
    width: int
    height: int
    calib_path: Path
    device: str
    session_id: Optional[str] = None


class L1Runtime:
    """infer + session.mjpeg + session_pose.jsonl + L3 post_process 标记。"""

    def __init__(
        self,
        l3_params: L3Params,
        *,
        fps: int = 60,
    ) -> None:
        self._l3_params = l3_params
        self._fps = max(1, int(fps))
        self._ring: EncodedRingBuffer | None = None
        self._mjpeg_session: SessionMjpegRecordSession | None = None
        self._session_meta: SessionMeta | None = None
        self._pose_log: L1PoseSessionLog | None = None
        self._last_pose_sample: Optional[Pose3DSample] = None

    @property
    def l3_params(self) -> L3Params:
        return self._l3_params

    @property
    def ring(self) -> EncodedRingBuffer | None:
        return self._ring

    @property
    def mjpeg_session(self) -> SessionMjpegRecordSession:
        if self._mjpeg_session is None:
            raise RuntimeError("L1Runtime MJPEG session not started")
        return self._mjpeg_session

    @property
    def ring_session(self) -> SessionMjpegRecordSession:
        return self.mjpeg_session

    def set_session_meta(self, meta: SessionMeta) -> None:
        self._session_meta = meta

    def start_session_mjpeg_recording(
        self,
        *,
        session_mjpeg_packets: int = 64,
        mjpeg_queue_depth: int = 64,
    ) -> None:
        """MJPEG session：ZED grab ingress。"""
        if self._mjpeg_session is not None:
            return
        meta = self._session_meta
        if meta is None:
            raise RuntimeError("session meta not set")
        clip_dir = self._l3_params.clip_dir
        if clip_dir is None:
            raise RuntimeError("session 录制需要 clip_dir（L3 输出目录）")

        session = create_session_mjpeg_record(
            session_dir=clip_dir,
            width=meta.width,
            height=meta.height,
            fps=self._fps,
            device=meta.device,
            calib_path=meta.calib_path,
            session_id=meta.session_id,
            mjpeg_queue_depth=max(8, int(mjpeg_queue_depth)),
            session_mjpeg_packets=max(16, int(session_mjpeg_packets)),
        )
        self._mjpeg_session = session
        self._ring = session.ring
        self._pose_log = session.pose_log

    def submit_mjpeg_ingress(self, capture_seq: int, data: bytes) -> bool:
        """mjpeg_tee ingress → session.mjpeg（非阻塞）。"""
        session = self._mjpeg_session
        if session is None:
            return False
        return session.submit_ingress_packet(int(capture_seq), data)

    def on_pose3d_frame(
        self,
        *,
        capture_seq: int,
        infer_seq: int,
        kpts3d: np.ndarray | None,
        tri_ok: bool | None = None,
        det_left: dict | None = None,
        det_right: dict | None = None,
    ) -> Pose3DSample:
        kpts_copy: Optional[np.ndarray] = None
        if kpts3d is not None:
            kpts_copy = np.asarray(kpts3d, dtype=np.float32).copy()
        sample = Pose3DSample(
            capture_seq=int(capture_seq),
            infer_seq=int(infer_seq),
            kpts3d=kpts_copy,
            tri_ok=bool(tri_ok) if tri_ok is not None else (kpts_copy is not None),
            det_left=_copy_det(det_left),
            det_right=_copy_det(det_right),
        )
        self._last_pose_sample = sample
        if self._pose_log is not None:
            self._pose_log.append_sample(sample)
        return sample

    def flush_pose_log(self) -> None:
        if self._pose_log is not None:
            self._pose_log.flush()

    def latest_pose3d_sample(self) -> Optional[Pose3DSample]:
        return self._last_pose_sample

    def snapshot_pose3d(self) -> list[Pose3DSample]:
        if self._last_pose_sample is None:
            return []
        return [self._last_pose_sample]

    def pose3d_history_len(self) -> int:
        if self._pose_log is not None:
            return self._pose_log.count
        return 1 if self._last_pose_sample is not None else 0

    def notify_segment_registered(
        self,
        start_capture_seq: int,
        end_capture_seq: int,
        *,
        label: Optional[str] = None,
        index: int,
    ) -> None:
        self._mark_post_process(
            start_capture_seq,
            end_capture_seq,
            index=index,
            label=label,
        )

    def _mark_post_process(
        self,
        start: int,
        end: int,
        *,
        index: int,
        label: Optional[str],
    ) -> None:
        reg = self._l3_params.clip_registry
        if reg is not None:
            reg.append_persist_status(
                index=int(index),
                start_capture_seq=int(start),
                end_capture_seq=int(end),
                status="post_process",
            )
        logger.info(
            "session clip marked post_process %s..%s label=%s",
            start,
            end,
            label,
        )

    def finalize_session_recording(self) -> dict:
        """session 模式：flush 写盘 + 写 session_meta stop 字段。"""
        session = self._mjpeg_session
        if session is None:
            return {}

        pose_count = self._pose_log.count if self._pose_log is not None else 0
        duration = session.recording_duration_sec()
        duration_wall = max(0.0, time.time() - float(session.started_ts))
        st = session.stop()
        rec_fps = (
            round(st.frames_submitted / duration, 2)
            if duration > 0 and st.frames_submitted > 0
            else 0.0
        )
        summary = {
            "frames_submitted": st.frames_submitted,
            "frames_recorded": st.h264_packets,
            "enc_dropped": st.enc_dropped,
            "mjpeg_bytes": st.h264_bytes,
            "mjpeg_packets": st.h264_packets,
            "pose_rows": pose_count,
            "duration_sec": round(duration, 3),
            "duration_wall_sec": round(duration_wall, 3),
            "rec_fps": rec_fps,
            "session_dir": str(session.bundle.session_dir),
        }
        finalize_session_meta(
            session.bundle.meta_path,
            frames_submitted=st.frames_submitted,
            frames_recorded=st.h264_packets,
            frames_pose=pose_count,
            duration_sec=duration,
            enc_dropped=st.enc_dropped,
            h264_bytes=st.h264_bytes,
            duration_wall_sec=duration_wall,
        )

        self._mjpeg_session = None
        self._ring = None
        self._pose_log = None
        return summary

    def release_ring_buffers(self) -> None:
        if self._ring is not None:
            self._ring.clear()

    def stop(self) -> None:
        if self._mjpeg_session is not None:
            self._mjpeg_session.stop()
            self._mjpeg_session = None
        if self._pose_log is not None:
            self._pose_log.close()
            self._pose_log = None
        self._ring = None

    def status_line(self, *, ring_stats: dict | None = None) -> str:
        if self._mjpeg_session is not None:
            st = self._mjpeg_session.stats()
            ring, extra = self._format_mjpeg_status(st)
        elif ring_stats is not None:
            ring, extra = self._format_mjpeg_status(ring_stats)
        else:
            ring = "session_mjpeg=未启动"
            extra = ""
        hist_len = self.pose3d_history_len()
        pose_tag = f"pose_log={hist_len}"
        return f"L1: {pose_tag}  {ring}{extra}"

    def _format_mjpeg_status(self, st: dict) -> tuple[str, str]:
        ring = f"session_mjpeg pkts={st.get('mjpeg_packets_pulled', '?')}"
        extra = (
            f" pkts={st.get('session_h264_packets', 0)} "
            f"mjpeg={st.get('session_h264_bytes', 0)}B"
            f" drop={st.get('session_writer_dropped', 0)}"
        )
        if st.get("frames_submitted") is not None:
            extra = f" submitted={st.get('frames_submitted', 0)}" + extra
        if self._pose_log is not None:
            extra += f" pose_log={self._pose_log.count}"
        return ring, extra
