"""运行时 clip 注册表：L3 识别动作后依次追加 manifest。"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from contracts.segment import Segment, load_segments_file


@dataclass
class ClipRegistry:
    manifest_path: Path
    _segments: list[Segment] = field(default_factory=list)
    _manifest_index: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def segments(self) -> list[Segment]:
        with self._lock:
            return list(self._segments)

    def append_segment(
        self,
        start_capture_seq: int,
        end_capture_seq: int,
        *,
        label: Optional[str] = None,
    ) -> int:
        start, end = int(start_capture_seq), int(end_capture_seq)
        if end < start:
            raise ValueError(
                f"end_capture_seq ({end}) must be >= start_capture_seq ({start})"
            )
        spec = Segment(start_capture_seq=start, end_capture_seq=end, label=label)
        with self._lock:
            idx = self._manifest_index
            self._manifest_index += 1
            self._segments.append(spec)
            self._append_manifest_line(
                {
                    "type": "segment",
                    "status": "detected",
                    "index": idx,
                    "start_capture_seq": start,
                    "end_capture_seq": end,
                    "label": label,
                    "ts": time.time(),
                }
            )
            return len(self._segments) - 1

    def append_marker(
        self,
        capture_seq: int,
        *,
        label: Optional[str] = None,
        kind: str = "marker",
    ) -> int:
        seq = int(capture_seq)
        with self._lock:
            idx = self._manifest_index
            self._manifest_index += 1
            self._append_manifest_line(
                {
                    "type": "marker",
                    "index": idx,
                    "capture_seq": seq,
                    "kind": str(kind),
                    "label": label,
                    "ts": time.time(),
                }
            )
            return idx

    def append_persist_status(
        self,
        *,
        index: int,
        start_capture_seq: int,
        end_capture_seq: int,
        status: str,
        nv12: Optional[str] = None,
        meta: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        obj: dict[str, Any] = {
            "type": "persist",
            "index": int(index),
            "start_capture_seq": int(start_capture_seq),
            "end_capture_seq": int(end_capture_seq),
            "status": str(status),
            "ts": time.time(),
        }
        if nv12 is not None:
            obj["nv12"] = nv12
        if meta is not None:
            obj["meta"] = meta
        if error is not None:
            obj["error"] = error
        with self._lock:
            self._append_manifest_line(obj)

    def _append_manifest_line(self, obj: dict[str, Any]) -> None:
        path = Path(self.manifest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(obj, ensure_ascii=False) + "\n")

    @classmethod
    def open_session(cls, clip_dir: Path) -> ClipRegistry:
        out = Path(clip_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        manifest = out / "clips_manifest.jsonl"
        if manifest.is_file():
            manifest.unlink()
        return cls(manifest_path=manifest)

    @classmethod
    def from_replay_file(cls, path: Path, *, clip_dir: Path) -> ClipRegistry:
        specs = load_segments_file(path)
        reg = cls.open_session(clip_dir)
        for spec in specs:
            reg.append_segment(
                spec.start_capture_seq,
                spec.end_capture_seq,
                label=spec.label,
            )
        return reg

    @classmethod
    def from_specs(cls, specs: list[Segment], *, clip_dir: Path) -> ClipRegistry:
        reg = cls.open_session(clip_dir)
        for spec in specs:
            reg.append_segment(
                spec.start_capture_seq,
                spec.end_capture_seq,
                label=spec.label,
            )
        return reg
