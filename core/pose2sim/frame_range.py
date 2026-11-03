"""jsonl 按 capture_seq 裁段（路径 6 手动区间）。"""

from __future__ import annotations

import json
from pathlib import Path


def filter_rows_by_capture_seq(
    rows: list[dict],
    *,
    start_seq: int | None = None,
    end_seq: int | None = None,
) -> list[dict]:
    """保留 start_seq <= capture_seq <= end_seq（含端点；None=不限）。"""
    out: list[dict] = []
    for row in rows:
        seq = int(row.get("capture_seq", -1))
        if start_seq is not None and seq < int(start_seq):
            continue
        if end_seq is not None and seq > int(end_seq):
            continue
        out.append(row)
    return out


def load_jsonl_rows(
    path: Path,
    *,
    start_seq: int | None = None,
    end_seq: int | None = None,
    max_frames: int = 0,
    require_kpts3d: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if require_kpts3d and "kpts3d" not in row:
                continue
            rows.append(row)
    rows = filter_rows_by_capture_seq(rows, start_seq=start_seq, end_seq=end_seq)
    if max_frames > 0:
        rows = rows[: int(max_frames)]
    return rows
