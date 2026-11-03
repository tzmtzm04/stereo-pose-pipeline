"""Pose2Sim TRC → tri_*_3d.jsonl（COCO-17 / 133×4）。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np

from core.pose2sim.coco17_layout import COCO17_P2S_MARKERS
from core.pose2sim.trc_export import (
    pose2sim_trc_yup_to_opencv_left,
    yup_xyz_to_opencv_left,
)

TrcYupLayout = Literal["scheme_b", "pose2sim_native"]


def _read_trc(trc_path: Path):
    """读 TRC（纯 Python，不依赖 Pose2Sim / pandas）。"""
    trc_path = Path(trc_path)
    lines = trc_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        raise ValueError(f"TRC too short: {trc_path}")
    header = lines[:5]
    markers = [m.strip() for m in header[3].split("\t")[2::3] if m.strip()]
    n_markers = len(markers)

    frames: list[int] = []
    times: list[float] = []
    coords: list[list[float]] = []

    for line in lines[5:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        need = 2 + n_markers * 3
        if len(parts) < need:
            continue
        frames.append(int(float(parts[0])))
        times.append(float(parts[1]))
        row_vals: list[float] = []
        for mi in range(n_markers):
            base = 2 + mi * 3
            for j in range(3):
                txt = parts[base + j].strip()
                row_vals.append(float(txt) if txt else float("nan"))
        coords.append(row_vals)

    if not coords:
        raise ValueError(f"TRC has no data rows: {trc_path}")

    col_names: list[str] = []
    for m in markers:
        col_names.extend([m, m, m])

    class _ColFrame:
        def __init__(self, data: np.ndarray, columns: list[str]):
            self._data = data
            self.columns = columns

        @property
        def iloc(self):
            return self

        def __getitem__(self, key):
            if isinstance(key, tuple):
                fi, col = key
                if isinstance(col, slice):
                    start = col.start or 0
                    stop = col.stop if col.stop is not None else self._data.shape[1]
                    return self._data[fi, start:stop]
                if isinstance(col, int):
                    return float(self._data[fi, col])
            raise KeyError(key)

        def __len__(self):
            return int(self._data.shape[0])

    q_coords = _ColFrame(np.asarray(coords, dtype=np.float64), col_names)
    frames_arr = np.asarray(frames, dtype=np.int64)

    class _Series:
        def __init__(self, values: np.ndarray):
            self._values = values

        @property
        def iloc(self):
            return self

        def __getitem__(self, idx: int):
            return int(self._values[idx])

        def __len__(self):
            return int(self._values.shape[0])

    frames_col = _Series(frames_arr)
    time_col = _Series(np.asarray(times, dtype=np.float64))
    return q_coords, frames_col, time_col, markers, header


def _trc_yup_to_opencv(
    x_yup: float,
    y_yup: float,
    z_yup: float,
    *,
    layout: TrcYupLayout,
) -> np.ndarray:
    if layout == "pose2sim_native":
        return pose2sim_trc_yup_to_opencv_left(x_yup, y_yup, z_yup)
    return yup_xyz_to_opencv_left(x_yup, y_yup, z_yup)


def apply_trc_to_rows(
    rows: list[dict],
    *,
    trc_path: Path,
    full_kpts: int = 133,
    pose2sim_tri: bool = True,
    pose2sim_filter3d: bool = False,
    trc_yup_layout: TrcYupLayout = "scheme_b",
) -> list[dict]:
    """TRC marker → 更新 template rows 的 kpts3d。"""
    Q_coords, frames_col, _time_col, markers, _header = _read_trc(trc_path)
    name_to_coco = {name: coco_i for name, coco_i in COCO17_P2S_MARKERS}

    # TRC Frame# → row index（导出时 frame_i 对应 rows[i]）
    row_by_fi: dict[int, dict] = {fi: row for fi, row in enumerate(rows)}
    row_by_seq: dict[int, dict] = {}
    for fi, row in enumerate(rows):
        row_by_seq[int(row.get("capture_seq", fi))] = row

    out_rows: list[dict] = []
    for fi in range(len(frames_col)):
        frame_id = int(frames_col.iloc[fi])
        row = row_by_fi.get(fi)
        if row is None:
            row = row_by_seq.get(frame_id)
        if row is None:
            continue
        rec = dict(row)
        raw = rec.get("kpts3d")
        if raw is None:
            kpts3d = np.full((full_kpts, 4), np.nan, dtype=np.float64)
            kpts3d[:, 3] = 0.0
        else:
            kpts3d = np.asarray(raw, dtype=np.float64)
            if kpts3d.ndim != 2:
                kpts3d = np.full((full_kpts, 4), np.nan, dtype=np.float64)
                kpts3d[:, 3] = 0.0
            elif kpts3d.shape[0] < full_kpts:
                pad = np.full((full_kpts, 4), np.nan, dtype=np.float64)
                pad[:, 3] = 0.0
                pad[: kpts3d.shape[0]] = kpts3d
                kpts3d = pad
            else:
                kpts3d = kpts3d.copy()

        for mi, marker in enumerate(markers):
            coco_i = name_to_coco.get(marker)
            if coco_i is None:
                continue
            base = mi * 3
            try:
                x = float(Q_coords.iloc[fi, base])
                y = float(Q_coords.iloc[fi, base + 1])
                z = float(Q_coords.iloc[fi, base + 2])
            except (IndexError, TypeError, ValueError):
                continue
            if not all(math.isfinite(v) for v in (x, y, z)):
                continue
            xyz = _trc_yup_to_opencv(x, y, z, layout=trc_yup_layout)
            if np.all(np.isfinite(xyz)):
                kpts3d[coco_i, :3] = xyz
                kpts3d[coco_i, 3] = 1.0

        rec["kpts3d"] = kpts3d.tolist()
        rec["tri_ok"] = bool(np.any(np.isfinite(kpts3d[:, :3])))
        rec["pose2sim_tri"] = bool(pose2sim_tri)
        rec["pose2sim_filter3d"] = bool(pose2sim_filter3d)
        rec["anipose_optim"] = False
        rec["anipose_filter3d"] = False
        out_rows.append(rec)

    written_seqs = {int(r.get("capture_seq", -1)) for r in out_rows}
    for row in rows:
        seq = int(row.get("capture_seq", -1))
        if seq not in written_seqs:
            rec = dict(row)
            rec.setdefault("pose2sim_tri", False)
            rec.setdefault("pose2sim_filter3d", False)
            out_rows.append(rec)
    out_rows.sort(key=lambda r: int(r.get("capture_seq", 0)))
    return out_rows


def write_trc_jsonl(
    *,
    template_rows: list[dict],
    trc_path: Path,
    out_path: Path,
    pose2sim_tri: bool = True,
    pose2sim_filter3d: bool = False,
    trc_yup_layout: TrcYupLayout = "scheme_b",
) -> int:
    out_rows = apply_trc_to_rows(
        template_rows,
        trc_path=trc_path,
        pose2sim_tri=pose2sim_tri,
        pose2sim_filter3d=pose2sim_filter3d,
        trc_yup_layout=trc_yup_layout,
    )
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        for row in out_rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(out_rows)
