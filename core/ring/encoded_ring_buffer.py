"""NVENC H.264 编码环形缓冲。"""

from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

from core.ring.h264_nal_util import packet_is_keyframe

DEFAULT_MAX_PACKETS = 2400


@dataclass(frozen=True)
class EncodedPacket:
    capture_seq: int
    pts_ns: int
    is_keyframe: bool
    data: bytes


@dataclass(frozen=True)
class EncodedClipRange:
    requested_start_capture_seq: int
    requested_end_capture_seq: int
    actual_start_capture_seq: int
    actual_end_capture_seq: int


class EncodedRingBuffer:
    """NVENC H.264 byte-stream 环形缓冲。"""

    def __init__(
        self,
        *,
        max_packets: int = DEFAULT_MAX_PACKETS,
        iframe_interval: int = 15,
    ) -> None:
        self._max_packets = max(1, int(max_packets))
        self._iframe_interval = max(1, int(iframe_interval))
        self._buffer: Deque[EncodedPacket] = deque()
        self._lock = threading.Lock()

    @property
    def max_packets(self) -> int:
        return self._max_packets

    def len(self) -> int:
        with self._lock:
            return len(self._buffer)

    def push(self, packet: EncodedPacket) -> None:
        with self._lock:
            self._buffer.append(packet)
            while len(self._buffer) > self._max_packets:
                self._buffer.popleft()

    def _is_keyframe(self, pkt: EncodedPacket) -> bool:
        return packet_is_keyframe(
            pkt.data,
            is_keyframe_flag=pkt.is_keyframe,
            capture_seq=pkt.capture_seq,
            iframe_interval=self._iframe_interval,
        )

    def find_clip(
        self,
        start_capture_seq: int,
        end_capture_seq: int,
    ) -> Optional[EncodedClipRange]:
        start = int(start_capture_seq)
        end = int(end_capture_seq)
        if end < start:
            return None

        with self._lock:
            packets = list(self._buffer)
        if not packets:
            return None

        seqs = [p.capture_seq for p in packets]
        min_seq, max_seq = seqs[0], seqs[-1]
        if end < min_seq or start > max_seq:
            return None

        actual_start: Optional[int] = None
        for pkt in packets:
            if pkt.capture_seq <= start and self._is_keyframe(pkt):
                actual_start = pkt.capture_seq
        if actual_start is None:
            for pkt in packets:
                if self._is_keyframe(pkt):
                    actual_start = pkt.capture_seq
                    break
        if actual_start is None:
            return None

        actual_end = max_seq
        for pkt in packets:
            if pkt.capture_seq > end and self._is_keyframe(pkt):
                actual_end = pkt.capture_seq - 1
                break

        if actual_end < actual_start:
            return None

        return EncodedClipRange(
            requested_start_capture_seq=start,
            requested_end_capture_seq=end,
            actual_start_capture_seq=actual_start,
            actual_end_capture_seq=actual_end,
        )

    def stats(self) -> dict:
        with self._lock:
            n = len(self._buffer)
            if n == 0:
                return {"packets": 0}
            seqs = [p.capture_seq for p in self._buffer]
            keyframes = sum(1 for p in self._buffer if self._is_keyframe(p))
            return {
                "packets": n,
                "capture_seq_min": seqs[0],
                "capture_seq_max": seqs[-1],
                "keyframes": keyframes,
            }

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
