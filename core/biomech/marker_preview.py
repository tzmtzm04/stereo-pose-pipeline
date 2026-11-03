"""BioModule v1 marker 预览工具。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from core.biomech.marker_layout import MARKER_COUNT, MARKER_MODEL_COUNT

MARKER_CRITERION = "dimensionless_marker"


def _marker_opensim_to_preview(pts: np.ndarray) -> np.ndarray:
    """OpenSim body (Y-up) → 左相机预览系 (Y-down, X 与 H36M 对齐)。"""
    out = np.asarray(pts, dtype=np.float32).copy()
    out[:, 0] *= -1.0
    out[:, 1] *= -1.0
    return out


def load_biomech_by_capture_seq(path: Path) -> dict[int, dict]:
    out: dict[int, dict] = {}
    with Path(path).open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seq = int(row["capture_seq"])
            out[seq] = row
    return out


def biomech_row_to_marker_kpts3d(row: dict | None) -> np.ndarray | None:
    """biomech.jsonl 行 → (19,4) marker 点（M1–M19），M1 居中。"""
    if not row:
        return None
    biomech = row.get("biomech") or {}
    vals = biomech.get(MARKER_CRITERION)
    if vals is None:
        return None
    arr = np.asarray(vals, dtype=np.float32).reshape(-1)
    if arr.size == MARKER_MODEL_COUNT * 3:
        pts = arr.reshape(MARKER_MODEL_COUNT, 3)[:MARKER_COUNT]
    elif arr.size == MARKER_COUNT * 3:
        pts = arr.reshape(MARKER_COUNT, 3)
    else:
        return None
    pts = pts - pts[0:1]
    pts = _marker_opensim_to_preview(pts)
    out = np.zeros((MARKER_COUNT, 4), dtype=np.float32)
    out[:, :3] = pts
    out[:, 3] = 1.0
    return out


def biomech_contact_summary(row: dict | None) -> tuple[float | None, float | None]:
    if not row:
        return None, None
    summary = row.get("biomech_summary") or {}
    touch = summary.get("touch")
    seat = summary.get("seat")
    return (
        float(touch) if touch is not None else None,
        float(seat) if seat is not None else None,
    )
