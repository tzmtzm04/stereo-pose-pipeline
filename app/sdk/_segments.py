"""session segments.json 读写（capture_seq 闭区间）。"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from contracts.segment import Segment, load_segments_file

SEGMENTS_FILENAME = "segments.json"


def segments_path(session_dir: Path) -> Path:
    return Path(session_dir).expanduser().resolve() / SEGMENTS_FILENAME


def load_session_segments(session_dir: Path) -> list[Segment]:
    path = segments_path(session_dir)
    if not path.is_file():
        return []
    return load_segments_file(path)


def write_segments_file(path: Path, segments: list[Segment]) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    clips = [
        {
            "label": s.label,
            "start_capture_seq": int(s.start_capture_seq),
            "end_capture_seq": int(s.end_capture_seq),
        }
        for s in segments
    ]
    path.write_text(
        json.dumps({"clips": clips}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


class SegmentTracker:
    """测评段 start/end → segments.json。"""

    def __init__(self, session_dir: Path) -> None:
        self._session_dir = Path(session_dir).expanduser().resolve()
        self._lock = threading.Lock()
        self._closed: list[Segment] = load_session_segments(self._session_dir)
        self._open: dict[str, int] = {}

    def mark(self, label: str, event: str, capture_seq: int) -> Segment | None:
        key = str(label).strip()
        if not key:
            raise ValueError("segment label 不能为空")
        seq = int(capture_seq)
        with self._lock:
            if event == "start":
                self._open[key] = seq
                return None
            if event != "end":
                raise ValueError(f"未知 segment event: {event!r}")
            start = self._open.pop(key, seq)
            seg = Segment(
                start_capture_seq=min(start, seq),
                end_capture_seq=max(start, seq),
                label=key,
            )
            self._closed.append(seg)
            self._closed.sort(key=lambda s: s.start_capture_seq)
            write_segments_file(segments_path(self._session_dir), self._closed)
            return seg

    def close_with_seqs(
        self,
        label: str,
        start_capture_seq: int,
        end_capture_seq: int,
    ) -> Segment:
        """算法识别边界：直接写入闭区间段（覆盖未闭合 start）。"""
        key = str(label).strip()
        if not key:
            raise ValueError("segment label 不能为空")
        start = int(start_capture_seq)
        end = int(end_capture_seq)
        if end < start:
            start, end = end, start
        with self._lock:
            self._open.pop(key, None)
            seg = Segment(
                start_capture_seq=start,
                end_capture_seq=end,
                label=key,
            )
            self._closed.append(seg)
            self._closed.sort(key=lambda s: s.start_capture_seq)
            write_segments_file(segments_path(self._session_dir), self._closed)
            return seg

    @property
    def segments(self) -> list[Segment]:
        with self._lock:
            return list(self._closed)
