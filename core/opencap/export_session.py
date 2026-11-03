"""full_session_pose3d.jsonl (WholeBody-133) → OpenCap PreAugmentation TRC。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.opencap.opencap20_layout import (
    NUM_OPENCAP20,
    OPENCAP20_MARKER_NAMES,
    wholebody133_to_opencap20,
)
from core.opencap.trc_io import left_cam_opencv_to_zup, write_trc, zup_to_yup

_MIN_CONF = 1e-4


def _opencap20_from_row(row: dict, *, min_conf: float) -> np.ndarray | None:
    """从 WholeBody-133 三角化结果映射 OpenCap-20（不在此阶段做 3D 滤波）。"""
    raw = row.get("kpts3d")
    if raw is None:
        return None
    k = np.asarray(raw, dtype=np.float32)
    if k.ndim != 2 or k.shape[0] < 23:
        return None
    return wholebody133_to_opencap20(k, min_conf=min_conf)


@dataclass(frozen=True)
class OpencapExportStats:
    frames_in: int
    frames_written: int
    trc_path: Path | None
    out_dir: Path | None


def _load_pose_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "kpts3d" in row:
                rows.append(row)
    return rows


def _frame_rate_from_meta(session_dir: Path, default: float = 60.0) -> float:
    for name in ("full_session_meta.json", "session_meta.json"):
        p = session_dir / name
        if not p.is_file():
            # also check offline meta next to pose file
            continue
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            fps = float(meta.get("fps") or meta.get("frame_rate") or default)
            if fps > 0:
                return fps
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return default


def export_opencap20_trc(
    pose_path: Path,
    *,
    session_dir: Path,
    out_dir: Path | None = None,
    frame_rate: float | None = None,
    min_conf: float = _MIN_CONF,
    tri_ok_only: bool = False,
) -> OpencapExportStats:
    pose_path = Path(pose_path).expanduser().resolve()
    session_dir = Path(session_dir).expanduser().resolve()
    if out_dir is None:
        out_dir = session_dir / "opencap"
    out_dir = Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_pose_rows(pose_path)
    if not rows:
        return OpencapExportStats(0, 0, None, out_dir)

    fps = float(frame_rate or _frame_rate_from_meta(session_dir))
    # offline meta may live beside pose
    offline_meta = pose_path.parent / "full_session_meta.json"
    if frame_rate is None and offline_meta.is_file():
        try:
            meta = json.loads(offline_meta.read_text(encoding="utf-8"))
            fps = float(meta.get("fps") or fps)
        except (json.JSONDecodeError, TypeError, ValueError):
            pass

    frames: list[int] = []
    zup_blocks: list[np.ndarray] = []

    for row in rows:
        if tri_ok_only and not row.get("tri_ok", True):
            continue
        oc20 = _opencap20_from_row(row, min_conf=min_conf)
        if oc20 is None:
            continue
        block = np.full(NUM_OPENCAP20 * 3, np.nan, dtype=np.float32)
        ok_any = False
        for mi in range(NUM_OPENCAP20):
            p = oc20[mi]
            if not np.all(np.isfinite(p[:3])) or float(p[3]) < min_conf:
                continue
            block[mi * 3 : mi * 3 + 3] = left_cam_opencv_to_zup(p[:3])
            ok_any = True
        if not ok_any:
            continue
        # midHip 必须有效才对 LSTM 有用
        mid = oc20[OPENCAP20_MARKER_NAMES.index("midHip")]
        if not np.all(np.isfinite(mid[:3])):
            continue
        frames.append(int(row.get("capture_seq", len(frames))))
        zup_blocks.append(block)

    if not frames:
        return OpencapExportStats(len(rows), 0, None, out_dir)

    zup_arr = np.stack(zup_blocks, axis=0)
    yup_flat = zup_to_yup(zup_arr)
    trc_path = out_dir / "PreAugmentation.trc"
    write_trc(
        trc_path,
        marker_names=OPENCAP20_MARKER_NAMES,
        frames=np.asarray(frames, dtype=np.int64),
        coords_yup=yup_flat,
        frame_rate=fps,
    )
    return OpencapExportStats(
        frames_in=len(rows),
        frames_written=len(frames),
        trc_path=trc_path,
        out_dir=out_dir,
    )
