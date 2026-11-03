"""MuJoCo IK qpos 时序 Butterworth 低通（零相位 filtfilt）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.geometry.anipose_filter import butterworth_lowpass_1d
from core.mujoco_ik.qpos_limits import clip_qpos_to_limits


@dataclass(frozen=True)
class QposButterworthConfig:
    fps: float = 60.0
    order: int = 4
    cutoff_hz: float = 6.0


@dataclass
class QposFilterStats:
    frames_in: int = 0
    nq: int = 0
    mean_abs_dq_before: float = 0.0
    mean_abs_dq_after: float = 0.0


def _read_ik_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r.get("capture_seq", -1)))
    return rows


def smooth_qpos_butterworth(
    qpos: np.ndarray,
    *,
    cfg: QposButterworthConfig,
) -> np.ndarray:
    """(T, nq) → 逐关节 Butterworth 低通。"""
    q = np.asarray(qpos, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] < 2:
        return q.copy()
    out = np.empty_like(q)
    for j in range(q.shape[1]):
        out[:, j] = butterworth_lowpass_1d(
            q[:, j],
            fps=float(cfg.fps),
            cutoff_hz=float(cfg.cutoff_hz),
            order=int(cfg.order),
        )
    return out


def _mean_step_delta(q: np.ndarray) -> float:
    if q.shape[0] < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(q, axis=0))))


def filter_mujoco_ik_jsonl_butterworth(
    in_path: Path,
    out_path: Path,
    *,
    cfg: QposButterworthConfig | None = None,
    model_path: Path | str | None = None,
) -> QposFilterStats:
    """path6_mujoco_ik.jsonl → 滤波后 jsonl（保留元数据，替换 qpos）。"""
    cfg = cfg or QposButterworthConfig()
    rows = _read_ik_rows(in_path)
    if not rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("", encoding="utf-8")
        return QposFilterStats()

    q_raw = np.stack([np.asarray(r["qpos"], dtype=np.float64) for r in rows], axis=0)
    q_filt = smooth_qpos_butterworth(q_raw, cfg=cfg)

    clip_model = None
    mujoco_mod = None
    if model_path is not None:
        import mujoco

        mujoco_mod = mujoco
        clip_model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row, qpos in zip(rows, q_filt):
            if clip_model is not None and mujoco_mod is not None:
                qpos = clip_qpos_to_limits(clip_model, qpos, mujoco_mod=mujoco_mod)
            rec = dict(row)
            rec["qpos"] = np.asarray(qpos, dtype=np.float64).tolist()
            rec["path6_mujoco_ik_filter"] = True
            rec["path6_mujoco_ik_filter_type"] = "butterworth"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return QposFilterStats(
        frames_in=len(rows),
        nq=int(q_raw.shape[1]),
        mean_abs_dq_before=_mean_step_delta(q_raw),
        mean_abs_dq_after=_mean_step_delta(q_filt),
    )
