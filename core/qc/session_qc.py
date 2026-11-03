"""离线 session QC：读 full_session_pose3d.jsonl → qc_scores.jsonl。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.geometry.calib_model import StereoCalibration
from core.qc.frame_scorer import QcConfig, QcFrameResult, _bone_lengths, score_pose_frame
from core.qc.h36m_joints import h36m_coord_for_export


@dataclass
class QcSessionStats:
    frames_in: int = 0
    frames_out: int = 0
    frames_pass: int = 0
    pass_rate: float = 0.0
    mean_score: float = 0.0


def _median_bone_ref(rows: list[dict], *, min_conf: float) -> dict[tuple[int, int], float]:
    acc: dict[tuple[int, int], list[float]] = {}
    for row in rows:
        if not row.get("tri_ok") or row.get("kpts3d") is None:
            continue
        k = np.asarray(row["kpts3d"], dtype=np.float32)
        for key, ln in _bone_lengths(k, min_conf=min_conf).items():
            acc.setdefault(key, []).append(ln)
    return {k: float(np.median(v)) for k, v in acc.items() if v}


def _result_to_row(base: dict, result: QcFrameResult) -> dict:
    out = {
        "capture_seq": int(base["capture_seq"]),
        "infer_seq": int(base.get("infer_seq", -1)),
        "tri_ok": bool(base.get("tri_ok", False)),
        "qc_score": round(float(result.qc_score), 4),
        "qc_pass": bool(result.qc_pass),
        "qc_flags": list(result.qc_flags),
        "mean_conf": round(float(result.mean_conf), 4),
        "valid_joint_count": int(result.valid_joint_count),
        "reproj_px": round(float(result.reproj_px), 3),
        "bone_outlier_ratio": round(float(result.bone_outlier_ratio), 4),
        "max_joint_speed_m_s": round(float(result.max_joint_speed_m_s), 3),
    }
    if result.coordinate is not None:
        out["coordinate"] = result.coordinate.tolist()
    if result.h36m_kpts is not None:
        out["h36m_kpts"] = result.h36m_kpts.tolist()
    if result.h36m_valid is not None:
        out["h36m_valid"] = [bool(x) for x in result.h36m_valid.tolist()]
    return out


def run_session_qc(
    in_path: Path,
    out_path: Path,
    *,
    cfg: QcConfig | None = None,
    calib: StereoCalibration | None = None,
    pass_only: bool = False,
) -> QcSessionStats:
    cfg = cfg or QcConfig()
    in_path = Path(in_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    with in_path.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            raw_rows.append(json.loads(line))

    bone_ref = _median_bone_ref(raw_rows, min_conf=cfg.min_conf)
    stats = QcSessionStats(frames_in=len(raw_rows))
    scores: list[float] = []
    prev_kpts: np.ndarray | None = None

    with out_path.open("w", encoding="utf-8") as out_fp:
        for base in raw_rows:
            result = score_pose_frame(
                base,
                cfg=cfg,
                calib=calib,
                bone_ref=bone_ref,
                prev_kpts3d=prev_kpts,
            )
            if base.get("kpts3d") is not None:
                prev_kpts = np.asarray(base["kpts3d"], dtype=np.float32)

            row = _result_to_row(base, result)
            scores.append(float(result.qc_score))
            stats.frames_out += 1
            if result.qc_pass:
                stats.frames_pass += 1
            if pass_only and not result.qc_pass:
                continue
            out_fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats.pass_rate = (
        float(stats.frames_pass / stats.frames_out) if stats.frames_out else 0.0
    )
    stats.mean_score = float(np.mean(scores)) if scores else 0.0
    return stats


def load_qc_pose_sequence(
    path: Path,
    *,
    pass_only: bool = False,
) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if pass_only and not row.get("qc_pass", False):
                continue
            rows.append(row)
    return rows


def qc_row_to_kpts3d_render(row: dict) -> np.ndarray | None:
    """预览用：优先 h36m_kpts，否则从 coordinate 还原为 (17,4)。"""
    if row.get("h36m_kpts") is not None:
        return np.asarray(row["h36m_kpts"], dtype=np.float32)
    return qc_row_to_h36m_input_render(row)


def qc_row_to_h36m_input_render(row: dict) -> np.ndarray | None:
    """BioModule 输入：pelvis 居中 coordinate (17,3) → (17,4)。"""
    if row.get("h36m_kpts") is not None:
        k = np.asarray(row["h36m_kpts"], dtype=np.float32)
        coord, valid = h36m_coord_for_export(k)
        out = np.zeros((17, 4), dtype=np.float32)
        n = min(17, coord.shape[0])
        out[:n, :3] = coord[:n]
        out[:n, 3] = valid[:n].astype(np.float32)
        return out

    coord = row.get("coordinate")
    if coord is None:
        return None
    c = np.asarray(coord, dtype=np.float32)
    if c.ndim != 2 or c.shape[0] < 17:
        return None
    out = np.zeros((17, 4), dtype=np.float32)
    out[:, :3] = c[:17]
    valid = row.get("h36m_valid")
    if valid is not None and len(valid) >= 17:
        for i, v in enumerate(valid[:17]):
            if v:
                out[i, 3] = 1.0
            else:
                out[i, :3] = 0.0
                out[i, 3] = 0.0
    else:
        out[:, 3] = 1.0
    return out
