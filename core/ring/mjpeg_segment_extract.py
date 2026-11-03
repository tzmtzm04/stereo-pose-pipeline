"""按 capture_seq 闭区间从 session.mjpeg 裁切运动段（重写 index offset）。"""

from __future__ import annotations

import json
from pathlib import Path

from core.offline.session_decoder import load_index_rows
from contracts.session_bundle import SessionBundle


def extract_mjpeg_segment(
    *,
    src_mjpeg: Path,
    src_index: Path,
    dst_mjpeg: Path,
    dst_index: Path,
    start_capture_seq: int,
    end_capture_seq: int,
) -> int:
    """复制 [start,end] 内 JPEG 包到新 mjpeg + session_index.jsonl；返回帧数。"""
    src_mjpeg = Path(src_mjpeg).expanduser().resolve()
    src_index = Path(src_index).expanduser().resolve()
    dst_mjpeg = Path(dst_mjpeg).expanduser().resolve()
    dst_index = Path(dst_index).expanduser().resolve()
    if not src_mjpeg.is_file() or not src_index.is_file():
        return 0

    bundle = SessionBundle.from_dir(src_mjpeg.parent)
    rows = load_index_rows(
        bundle,
        start_capture_seq=int(start_capture_seq),
        end_capture_seq=int(end_capture_seq),
    )
    if not rows:
        return 0

    dst_mjpeg.parent.mkdir(parents=True, exist_ok=True)
    offset = 0
    with src_mjpeg.open("rb") as src_fp, dst_mjpeg.open("wb") as dst_fp, dst_index.open(
        "w", encoding="utf-8"
    ) as idx_fp:
        for row in rows:
            src_fp.seek(int(row.offset))
            data = src_fp.read(int(row.size))
            if not data:
                continue
            dst_fp.write(data)
            idx_fp.write(
                json.dumps(
                    {
                        "offset": offset,
                        "size": len(data),
                        "capture_seq": int(row.capture_seq),
                        "pts_ns": int(row.pts_ns),
                        "is_keyframe": bool(row.is_keyframe),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            offset += len(data)
    return len(rows)
