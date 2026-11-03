"""动作段区间（capture_seq 闭区间）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass(frozen=True)
class Segment:
    """单个动作区间。"""

    start_capture_seq: int
    end_capture_seq: int
    label: Optional[str] = None

    def stem(self, index: int) -> str:
        base = f"clip_{index:03d}_{self.start_capture_seq}_{self.end_capture_seq}"
        if self.label:
            safe = "".join(
                c if c.isalnum() or c in "-_" else "_"
                for c in str(self.label)
            ).strip("_")
            if safe:
                return f"{base}_{safe}"
        return base


# 与上一版 ClipSpec 同义
ClipSpec = Segment


def _segment_from_obj(obj: Any, index: int) -> Segment:
    if not isinstance(obj, dict):
        raise ValueError(f"clips[{index}] must be object")
    start = obj.get("start", obj.get("start_capture_seq"))
    end = obj.get("end", obj.get("end_capture_seq"))
    if start is None or end is None:
        raise ValueError(
            f"clips[{index}] needs start/end or start_capture_seq/end_capture_seq"
        )
    start_i, end_i = int(start), int(end)
    if end_i < start_i:
        raise ValueError(f"clips[{index}] end ({end_i}) < start ({start_i})")
    label = obj.get("label") or obj.get("name")
    return Segment(
        start_capture_seq=start_i,
        end_capture_seq=end_i,
        label=str(label) if label is not None else None,
    )


def load_segments_file(path: Path) -> list[Segment]:
    """从 JSON 加载多段 clip，按 start_capture_seq 升序。"""
    p = Path(path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict) and "clips" in data:
        items = data["clips"]
    else:
        raise ValueError(f"{p}: expected {{clips:[...]}} or array")
    if not items:
        raise ValueError(f"{p}: clips is empty")
    segments = [_segment_from_obj(it, i) for i, it in enumerate(items)]
    return sorted(segments, key=lambda s: s.start_capture_seq)


def segment_from_single(start: int, end: int) -> list[Segment]:
    return [Segment(start_capture_seq=int(start), end_capture_seq=int(end))]
