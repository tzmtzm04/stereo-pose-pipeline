"""Stage 1：2D 清洗（Anipose medfilt 流程，参数可配）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import signal

from core.geometry.anipose_constraints import default_joint_indices
from core.geometry.anipose_filter import AniposeFilter2DConfig, filter_pose2d_stereo
from core.geometry.anipose_session import load_pose2d_from_jsonl
from core.geometry.one_euro import OneEuro2DConfig, filter_pose2d_stereo_one_euro
from core.offline.session_segment import (
    DEFAULT_OFFLINE_SEGMENT,
    OfflineSessionSegment,
    apply_segment_pose_stack,
)


@dataclass(frozen=True)
class CleanPose2DConfig:
    """ViTPose++ 2D 清洗默认（流程同 Anipose medfilt，阈值按 ViTPose conf 标定）。"""

    joint_set: str = "body"
    medfilt: int = 9
    offset_threshold: float = 25.0
    score_threshold: float = 0.25
    spline: bool = True
    min_valid_frac: float = 0.5
    min_valid_count: int = 5
    max_interp_gap: int = 0
    post_medfilt: int = 0
    one_euro: OneEuro2DConfig = OneEuro2DConfig(enabled=False)

    def to_filter_config(self) -> AniposeFilter2DConfig:
        return AniposeFilter2DConfig(
            enabled=True,
            medfilt=int(self.medfilt),
            offset_threshold=float(self.offset_threshold),
            score_threshold=float(self.score_threshold),
            spline=bool(self.spline),
            min_valid_frac=float(self.min_valid_frac),
            min_valid_count=int(self.min_valid_count),
            max_interp_gap=int(self.max_interp_gap),
        )


@dataclass
class CleanPose2DStats:
    frames_in_raw: int = 0
    frames_in: int = 0
    n_joints: int = 0
    n_cameras: int = 2
    valid_before: int = 0
    valid_after: int = 0
    one_euro_shift_mean_px: float = 0.0
    one_euro_shift_max_px: float = 0.0


def resolve_joint_indices(joint_set: str, n_kpts: int) -> tuple[int, ...]:
    key = str(joint_set).strip().lower().replace("-", "_")
    if key in ("wholebody", "wb133", "all", "133"):
        return tuple(range(n_kpts))
    if key in ("body_feet", "body+feet", "23", "bf"):
        return default_joint_indices("body_feet")
    if key in ("body", "17", "coco17"):
        return default_joint_indices("body")
    if key in ("body_limbs", "limbs", "no_head", "body12"):
        return default_joint_indices("body_limbs")
    raise ValueError(
        f"未知 joint_set: {joint_set!r}（wholebody | body_feet | body | body_limbs）"
    )


def _count_valid(points: np.ndarray, scores: np.ndarray, score_threshold: float) -> int:
    n_frames, n_joints, _ = points.shape
    total = 0
    for ji in range(n_joints):
        xy_ok = np.all(np.isfinite(points[:, ji, :]), axis=1)
        sc_ok = scores[:, ji] >= float(score_threshold)
        total += int(np.sum(xy_ok & sc_ok))
    return total


def _apply_post_medfilt(points: np.ndarray, kernel: int) -> np.ndarray:
    """对清洗后有限轨迹做时序 medfilt（Anipose 默认不平滑输出坐标，此处补一层）。"""
    k = int(kernel)
    if k <= 1:
        return points
    if k % 2 == 0:
        k += 1

    out = np.asarray(points, dtype=np.float64).copy()
    n_frames, n_joints, _ = out.shape
    for ji in range(n_joints):
        for axis in range(2):
            vals = out[:, ji, axis]
            ok = np.isfinite(vals)
            if int(np.sum(ok)) < k:
                continue
            padded = vals.copy()
            padded[~ok] = float(np.nanmedian(vals[ok]))
            smoothed = signal.medfilt(padded, kernel_size=k)
            vals[ok] = smoothed[ok]
            out[:, ji, axis] = vals
    return out


def _write_kpts_from_stack(
    rows: list[dict],
    points: np.ndarray,
    scores: np.ndarray,
    joint_indices: tuple[int, ...],
) -> list[dict]:
    out_rows: list[dict] = []
    n_frames = len(rows)
    for fi in range(n_frames):
        rec = dict(rows[fi])
        for ci, key in enumerate(("det_left_kpts", "det_right_kpts")):
            raw = rec.get(key)
            if raw is None:
                continue
            kpts = np.asarray(raw, dtype=np.float64).copy()
            if kpts.ndim != 2 or kpts.shape[1] < 3:
                continue
            for ji, src_j in enumerate(joint_indices):
                if src_j < 0 or src_j >= kpts.shape[0]:
                    continue
                xy = points[ci, fi, ji]
                sc = float(scores[ci, fi, ji])
                if np.all(np.isfinite(xy)):
                    kpts[src_j, 0] = xy[0]
                    kpts[src_j, 1] = xy[1]
                    kpts[src_j, 2] = sc
                else:
                    kpts[src_j, 0] = float("nan")
                    kpts[src_j, 1] = float("nan")
                    kpts[src_j, 2] = sc
            rec[key] = kpts.tolist()
        rec["clean_2d"] = True
        out_rows.append(rec)
    return out_rows


def run_clean_pose2d(
    *,
    pose_path: Path,
    out_path: Path,
    cfg: CleanPose2DConfig | None = None,
    max_frames: int = 0,
    segment: OfflineSessionSegment | None = None,
) -> tuple[CleanPose2DStats, dict]:
    cfg = cfg or CleanPose2DConfig()
    pose_path = Path(pose_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()

    raw_count = sum(1 for _ in pose_path.open(encoding="utf-8"))

    n_kpts = 0
    with pose_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            for key in ("det_left_kpts", "det_right_kpts"):
                raw = row.get(key)
                if raw is None:
                    continue
                arr = np.asarray(raw)
                if arr.ndim == 2 and arr.shape[0] > 0:
                    n_kpts = max(n_kpts, int(arr.shape[0]))
            if n_kpts > 0:
                break
    if n_kpts <= 0:
        raise ValueError(f"无 det_left/right_kpts: {pose_path}")

    joint_indices = resolve_joint_indices(cfg.joint_set, n_kpts)
    points, scores, rows = load_pose2d_from_jsonl(
        pose_path,
        joint_indices=joint_indices,
        max_frames=max_frames,
    )
    points, scores, rows, seg_used = apply_segment_pose_stack(
        points, scores, rows, segment
    )
    rows_count = points.shape[1]

    fcfg = cfg.to_filter_config()
    valid_before = _count_valid(points[0], scores[0], fcfg.score_threshold)
    valid_before += _count_valid(points[1], scores[1], fcfg.score_threshold)

    points_clean, scores_clean = filter_pose2d_stereo(points, scores, fcfg)
    if int(cfg.post_medfilt) > 1:
        for ci in range(points_clean.shape[0]):
            points_clean[ci] = _apply_post_medfilt(points_clean[ci], cfg.post_medfilt)

    one_euro_shifts: list[float] = []
    if cfg.one_euro.enabled:
        before_euro = np.asarray(points_clean, dtype=np.float64).copy()
        points_clean = filter_pose2d_stereo_one_euro(
            points_clean, scores_clean, cfg.one_euro
        )
        for ci in range(points_clean.shape[0]):
            b = before_euro[ci]
            a = points_clean[ci]
            valid = np.isfinite(b).all(axis=-1) & np.isfinite(a).all(axis=-1)
            if np.any(valid):
                d = np.linalg.norm(a[valid] - b[valid], axis=-1)
                one_euro_shifts.extend(d.tolist())

    valid_after = _count_valid(points_clean[0], scores_clean[0], fcfg.score_threshold)
    valid_after += _count_valid(points_clean[1], scores_clean[1], fcfg.score_threshold)

    out_rows = _write_kpts_from_stack(rows, points_clean, scores_clean, joint_indices)
    if cfg.one_euro.enabled:
        for row in out_rows:
            row["filter_2d"] = "anipose_medfilt+one_euro"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = CleanPose2DStats(
        frames_in_raw=raw_count,
        frames_in=rows_count,
        n_joints=len(joint_indices),
        valid_before=valid_before,
        valid_after=valid_after,
    )
    if one_euro_shifts:
        stats.one_euro_shift_mean_px = float(np.mean(one_euro_shifts))
        stats.one_euro_shift_max_px = float(np.max(one_euro_shifts))
    pipeline = "anipose_medfilt"
    if cfg.one_euro.enabled:
        pipeline += "+one_euro"
    summary = {
        "input": str(pose_path),
        "output": str(out_path),
        "pipeline": pipeline,
        "segment": seg_used.to_dict() if seg_used is not None else None,
        "config": asdict(cfg),
        "stats": asdict(stats),
    }
    return stats, summary


def write_clean_pose2d_summary(summary_path: Path, summary: dict) -> None:
    summary_path = Path(summary_path).expanduser().resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
