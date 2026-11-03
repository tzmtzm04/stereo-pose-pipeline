"""Session MJPEG + index → 逐帧 BGR（SBS 或左/右眼）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from contracts.session_bundle import SessionBundle


@dataclass(frozen=True)
class SessionIndexRow:
    offset: int
    size: int
    capture_seq: int
    pts_ns: int = 0
    is_keyframe: bool = True


@dataclass(frozen=True)
class SessionFrame:
    capture_seq: int
    sbs_bgr: np.ndarray
    left_bgr: np.ndarray
    right_bgr: np.ndarray


def load_session_meta(session_dir: Path) -> dict:
    meta_path = Path(session_dir) / "session_meta.json"
    if not meta_path.is_file():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_index_rows(
    bundle: SessionBundle,
    *,
    start_capture_seq: int | None = None,
    end_capture_seq: int | None = None,
) -> list[SessionIndexRow]:
    rows: list[SessionIndexRow] = []
    with bundle.index_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seq = int(obj["capture_seq"])
            if start_capture_seq is not None and seq < int(start_capture_seq):
                continue
            if end_capture_seq is not None and seq > int(end_capture_seq):
                continue
            rows.append(
                SessionIndexRow(
                    offset=int(obj["offset"]),
                    size=int(obj["size"]),
                    capture_seq=seq,
                    pts_ns=int(obj.get("pts_ns", 0)),
                    is_keyframe=bool(obj.get("is_keyframe", True)),
                )
            )
    return rows


def split_sbs_bgr(sbs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = sbs.shape[:2]
    if w <= h * 2:
        return sbs, sbs
    mid = w // 2
    return sbs[:, :mid].copy(), sbs[:, mid:].copy()


def decode_session_frame(
    bundle: SessionBundle,
    row: SessionIndexRow,
) -> SessionFrame | None:
    """按 index 行解码单帧 JPEG。"""
    with bundle.mjpeg_path.open("rb") as fp:
        fp.seek(int(row.offset))
        data = fp.read(int(row.size))
    if not data:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    sbs = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if sbs is None:
        return None
    left, right = split_sbs_bgr(sbs)
    return SessionFrame(
        capture_seq=int(row.capture_seq),
        sbs_bgr=sbs,
        left_bgr=left,
        right_bgr=right,
    )


def iter_session_frames(
    bundle: SessionBundle,
    *,
    start_capture_seq: int | None = None,
    end_capture_seq: int | None = None,
    stride: int = 1,
) -> Iterator[SessionFrame]:
    """按 index 顺序解码 session.mjpeg。"""
    rows = load_index_rows(
        bundle,
        start_capture_seq=start_capture_seq,
        end_capture_seq=end_capture_seq,
    )
    if int(stride) > 1:
        rows = rows[:: int(stride)]

    mjpeg_path = bundle.mjpeg_path
    with mjpeg_path.open("rb") as fp:
        for row in rows:
            fp.seek(int(row.offset))
            data = fp.read(int(row.size))
            if not data:
                continue
            arr = np.frombuffer(data, dtype=np.uint8)
            sbs = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if sbs is None:
                continue
            left, right = split_sbs_bgr(sbs)
            yield SessionFrame(
                capture_seq=int(row.capture_seq),
                sbs_bgr=sbs,
                left_bgr=left,
                right_bgr=right,
            )
