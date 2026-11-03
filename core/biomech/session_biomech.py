"""BioModule v1 批处理：qc_scores.jsonl → biomech.jsonl。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.biomech.runtime import (
    infer_session_poses,
    load_biomodule_v1,
    load_norm_stats,
    scalar_summary,
)
from core.qc.session_qc import load_qc_pose_sequence


@dataclass
class BiomechSessionStats:
    frames_in: int = 0
    frames_out: int = 0
    criteria: list[str] | None = None


def _row_coordinate(row: dict) -> np.ndarray | None:
    coord = row.get("coordinate")
    if coord is None:
        return None
    arr = np.asarray(coord, dtype=np.float32)
    if arr.shape != (17, 3):
        return None
    return arr


def run_session_biomech(
    qc_path: Path,
    out_path: Path,
    *,
    checkpoint: Path,
    norm_stats_path: Path,
    device: str = "cpu",
    batch_size: int = 64,
    pass_only: bool = False,
    require_qc_pass: bool = False,
) -> BiomechSessionStats:
    qc_path = Path(qc_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = load_qc_pose_sequence(qc_path, pass_only=False)
    stats = BiomechSessionStats(frames_in=len(rows))
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return stats

    coords: list[np.ndarray] = []
    valid_mask: list[bool] = []
    for row in rows:
        c = _row_coordinate(row)
        ok = c is not None
        valid_mask.append(ok)
        coords.append(c if ok else np.zeros((17, 3), dtype=np.float32))

    coord_seq = np.stack(coords, axis=0)
    stats_obj = load_norm_stats(norm_stats_path)
    model = load_biomodule_v1(checkpoint, device=device)
    biomech_rows = infer_session_poses(
        model,
        coord_seq,
        stats_obj,
        device=device,
        batch_size=batch_size,
    )

    with out_path.open("w", encoding="utf-8") as fp:
        for base, bio, ok in zip(rows, biomech_rows, valid_mask):
            if pass_only and not base.get("qc_pass", False):
                continue
            if require_qc_pass and not base.get("qc_pass", False):
                continue
            if not ok:
                continue
            summary = scalar_summary(bio)
            out = {
                "capture_seq": int(base["capture_seq"]),
                "infer_seq": int(base.get("infer_seq", -1)),
                "qc_pass": bool(base.get("qc_pass", False)),
                "qc_score": float(base.get("qc_score", 0.0)),
                "biomech_version": "v1",
                "biomech": bio,
                "biomech_summary": summary,
            }
            fp.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats.frames_out += 1

    stats.criteria = list(stats_obj.criteria)
    return stats
