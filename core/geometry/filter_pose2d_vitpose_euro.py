"""离线 Stage 1：Viterbi → Anipose medfilt → One Euro（可选第三层）。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from core.geometry.anipose_filter import filter_pose2d_stereo
from core.geometry.anipose_session import _dedupe_rows_by_capture_seq
from core.geometry.clean_pose2d import (
    CleanPose2DConfig,
    _write_kpts_from_stack,
    resolve_joint_indices,
)
from core.geometry.one_euro import OneEuro2DConfig, filter_pose2d_stereo_one_euro
from core.geometry.viterbi_pose2d import ViterbiPose2DConfig, filter_pose2d_viterbi_stereo
from core.offline.session_segment import OfflineSessionSegment


@dataclass(frozen=True)
class VitPose2DFilterConfig:
    """Stage 1：Viterbi（滤波1）+ medfilt（滤波2）+ One Euro（滤波3，默认开）。"""

    joint_set: str = "body"
    viterbi: ViterbiPose2DConfig = field(default_factory=ViterbiPose2DConfig)
    medfilt: CleanPose2DConfig = field(default_factory=CleanPose2DConfig)
    one_euro: OneEuro2DConfig = field(default_factory=OneEuro2DConfig)


# 兼容旧名
VitPoseEuro2DConfig = VitPose2DFilterConfig


@dataclass
class VitPose2DFilterStats:
    frames_in_raw: int = 0
    frames_in: int = 0
    frames_out: int = 0
    viterbi_shift_mean_px: float = 0.0
    viterbi_shift_max_px: float = 0.0
    medfilt_shift_mean_px: float = 0.0
    medfilt_shift_max_px: float = 0.0
    one_euro_shift_mean_px: float = 0.0
    one_euro_shift_max_px: float = 0.0


VitPoseEuro2DStats = VitPose2DFilterStats


def _apply_segment_rows(
    rows: list[dict],
    segment: OfflineSessionSegment | None,
) -> tuple[list[dict], OfflineSessionSegment | None]:
    if segment is None:
        return rows, None
    sl = segment.slice(len(rows))
    return rows[sl], segment


def _probe_n_kpts(rows: list[dict]) -> int:
    for row in rows:
        for key in ("det_left_kpts", "det_right_kpts"):
            raw = row.get(key)
            if raw is None:
                continue
            arr = np.asarray(raw)
            if arr.ndim == 2 and arr.shape[0] > 0:
                return int(arr.shape[0])
    return 0


def _hypo_key(kpts_key: str) -> str:
    if kpts_key.endswith("_kpts"):
        return kpts_key.replace("_kpts", "_kpts_hypo")
    return f"{kpts_key}_hypo"


def _stack_pose2d_from_rows(
    rows: list[dict],
    joint_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """rows → C×N×J×(×P)×2；优先读 *_kpts_hypo。"""
    n_frames = len(rows)
    n_joints = len(joint_indices)
    n_cams = 2
    top_p = 1
    for row in rows:
        for key in ("det_left_kpts", "det_right_kpts"):
            raw_h = row.get(_hypo_key(key))
            if raw_h is None:
                continue
            arr = np.asarray(raw_h, dtype=np.float64)
            if arr.ndim == 3 and arr.shape[-1] >= 3:
                top_p = max(top_p, int(arr.shape[1]))

    if top_p > 1:
        points = np.full((n_cams, n_frames, n_joints, top_p, 2), np.nan, dtype=np.float64)
        scores = np.zeros((n_cams, n_frames, n_joints, top_p), dtype=np.float64)
    else:
        points = np.full((n_cams, n_frames, n_joints, 2), np.nan, dtype=np.float64)
        scores = np.zeros((n_cams, n_frames, n_joints), dtype=np.float64)

    for fi, row in enumerate(rows):
        for ci, key in enumerate(("det_left_kpts", "det_right_kpts")):
            hypo_key = _hypo_key(key)
            raw_h = row.get(hypo_key)
            if raw_h is not None:
                kpts = np.asarray(raw_h, dtype=np.float64)
                if kpts.ndim != 3 or kpts.shape[-1] < 3:
                    continue
                for ji, src_j in enumerate(joint_indices):
                    if src_j < 0 or src_j >= kpts.shape[0]:
                        continue
                    for pi in range(min(top_p, kpts.shape[1])):
                        conf = float(kpts[src_j, pi, 2])
                        if top_p <= 1:
                            scores[ci, fi, ji] = conf
                        else:
                            scores[ci, fi, ji, pi] = conf
                        xy = kpts[src_j, pi, :2]
                        if np.all(np.isfinite(xy)):
                            if top_p <= 1:
                                points[ci, fi, ji] = xy
                            else:
                                points[ci, fi, ji, pi] = xy
                continue
            raw = row.get(key)
            if raw is None:
                continue
            kpts = np.asarray(raw, dtype=np.float64)
            if kpts.ndim != 2 or kpts.shape[1] < 2:
                continue
            for ji, src_j in enumerate(joint_indices):
                if src_j < 0 or src_j >= kpts.shape[0]:
                    continue
                conf = float(kpts[src_j, 2]) if kpts.shape[1] > 2 else 1.0
                xy = kpts[src_j, :2]
                if top_p <= 1:
                    scores[ci, fi, ji] = conf
                    if np.all(np.isfinite(xy)):
                        points[ci, fi, ji] = xy
                else:
                    scores[ci, fi, ji, 0] = conf
                    if np.all(np.isfinite(xy)):
                        points[ci, fi, ji, 0] = xy
    return points, scores


def _collapse_to_nj2(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim == 5:
        return arr[:, :, :, 0, :]
    if arr.ndim == 4 and arr.shape[-1] == 2:
        a, b, c, _ = arr.shape
        # C×N×J×2（C 小、J 大）原样；N×J×P×2 取 P=0
        if a <= 4 and c >= 10:
            return arr
        return arr[:, :, 0, :]
    return arr


def _collapse_scores(points: np.ndarray, scores: np.ndarray) -> np.ndarray:
    sc = np.asarray(scores, dtype=np.float64)
    if sc.ndim == 4:
        return sc[:, :, :, 0]
    if sc.ndim == 3:
        return sc
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 5 and sc.ndim == 4:
        return sc[:, :, :, 0]
    return sc


def _shift_stats(before: np.ndarray, after: np.ndarray) -> tuple[float, float, list[float]]:
    b = _collapse_to_nj2(before) if before.ndim > 3 else np.asarray(before, dtype=np.float64)
    a = _collapse_to_nj2(after) if after.ndim > 3 else np.asarray(after, dtype=np.float64)
    valid = np.isfinite(b).all(axis=-1) & np.isfinite(a).all(axis=-1)
    if not np.any(valid):
        return 0.0, 0.0, []
    d = np.linalg.norm(a[valid] - b[valid], axis=-1)
    return float(np.mean(d)), float(np.max(d)), d.tolist()


def _tag_rows(rows: list[dict], *, filter_2d: str, clean: bool) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        rec = dict(row)
        for side in ("left", "right"):
            rec.pop(f"det_{side}_kpts_hypo", None)
        rec["filter_2d"] = filter_2d
        rec["clean_2d"] = bool(clean)
        out.append(rec)
    return out


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_filter_pose2d_vitpose(
    *,
    pose_path: Path,
    out_path: Path,
    viterbi_out_path: Path | None = None,
    cfg: VitPose2DFilterConfig | None = None,
    segment: OfflineSessionSegment | None = None,
    max_frames: int = 0,
) -> tuple[VitPose2DFilterStats, dict]:
    cfg = cfg or VitPose2DFilterConfig()
    pose_path = Path(pose_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()

    raw_rows: list[dict] = []
    with pose_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                raw_rows.append(json.loads(line))
    rows = _dedupe_rows_by_capture_seq(raw_rows)
    if max_frames > 0:
        rows = rows[: int(max_frames)]
    rows, seg_used = _apply_segment_rows(rows, segment)

    n_kpts = _probe_n_kpts(rows)
    if n_kpts <= 0:
        raise ValueError(f"无 det_left/right_kpts: {pose_path}")

    joint_indices = resolve_joint_indices(cfg.joint_set, n_kpts)
    points, scores = _stack_pose2d_from_rows(rows, joint_indices)

    viterbi_shifts: list[float] = []
    rows_after_viterbi = rows

    if cfg.viterbi.enabled:
        before_v = np.asarray(points, dtype=np.float64).copy()
        points, scores = filter_pose2d_viterbi_stereo(points, scores, cfg.viterbi)
        for ci in range(points.shape[0]):
            _, _, ds = _shift_stats(before_v[ci], points[ci])
            viterbi_shifts.extend(ds)
        rows_after_viterbi = _write_kpts_from_stack(
            rows, points, _collapse_scores(points, scores), joint_indices
        )
        viterbi_path = (
            Path(viterbi_out_path).expanduser().resolve()
            if viterbi_out_path is not None
            else out_path.with_name("viterbi_2d.jsonl")
        )
        _write_jsonl(
            _tag_rows(rows_after_viterbi, filter_2d="viterbi", clean=False),
            viterbi_path,
        )
    else:
        viterbi_path = None

    pts2 = _collapse_to_nj2(points)
    sc2 = _collapse_scores(points, scores)
    before_m = np.asarray(pts2, dtype=np.float64).copy()
    pts_f, sc_f = filter_pose2d_stereo(
        pts2,
        sc2,
        cfg.medfilt.to_filter_config(),
    )
    medfilt_shifts: list[float] = []
    for ci in range(pts_f.shape[0]):
        _, _, ds = _shift_stats(before_m[ci], pts_f[ci])
        medfilt_shifts.extend(ds)

    rows_medfilt = _write_kpts_from_stack(
        rows_after_viterbi, pts_f, sc_f, joint_indices
    )
    medfilt_path: Path | None = None
    if cfg.one_euro.enabled:
        medfilt_path = out_path.with_name("medfilt_2d.jsonl")
        _write_jsonl(
            _tag_rows(rows_medfilt, filter_2d="viterbi+medfilt", clean=False),
            medfilt_path,
        )

    one_euro_shifts: list[float] = []
    pts_out = pts_f
    if cfg.one_euro.enabled:
        before_e = np.asarray(pts_f, dtype=np.float64).copy()
        pts_out = filter_pose2d_stereo_one_euro(pts_f, sc_f, cfg.one_euro)
        for ci in range(pts_out.shape[0]):
            _, _, ds = _shift_stats(before_e[ci], pts_out[ci])
            one_euro_shifts.extend(ds)

    filter_tag = "viterbi+medfilt"
    if cfg.one_euro.enabled:
        filter_tag += "+one_euro"
    rows_clean = _write_kpts_from_stack(
        rows_medfilt if cfg.one_euro.enabled else rows_after_viterbi,
        pts_out,
        sc_f,
        joint_indices,
    )
    _write_jsonl(
        _tag_rows(rows_clean, filter_2d=filter_tag, clean=True),
        out_path,
    )

    stats = VitPose2DFilterStats(
        frames_in_raw=len(raw_rows),
        frames_in=len(rows),
        frames_out=len(rows_clean),
    )
    if viterbi_shifts:
        stats.viterbi_shift_mean_px = float(np.mean(viterbi_shifts))
        stats.viterbi_shift_max_px = float(np.max(viterbi_shifts))
    if medfilt_shifts:
        stats.medfilt_shift_mean_px = float(np.mean(medfilt_shifts))
        stats.medfilt_shift_max_px = float(np.max(medfilt_shifts))
    if one_euro_shifts:
        stats.one_euro_shift_mean_px = float(np.mean(one_euro_shifts))
        stats.one_euro_shift_max_px = float(np.max(one_euro_shifts))

    parts = ["vitpose"]
    if cfg.viterbi.enabled:
        parts.append("viterbi")
    parts.append("medfilt")
    if cfg.one_euro.enabled:
        parts.append("one_euro")
    pipeline = "_".join(parts)
    summary = {
        "input": str(pose_path),
        "output": str(out_path),
        "viterbi_out": str(viterbi_path) if viterbi_path is not None else None,
        "medfilt_out": str(medfilt_path) if medfilt_path is not None else None,
        "pipeline": pipeline,
        "segment": seg_used.to_dict() if seg_used is not None else None,
        "config": {
            "joint_set": cfg.joint_set,
            "viterbi": asdict(cfg.viterbi),
            "medfilt": asdict(cfg.medfilt),
            "one_euro": asdict(cfg.one_euro),
        },
        "stats": asdict(stats),
    }
    return stats, summary


def run_filter_pose2d_vitpose_euro(
    *,
    pose_path: Path,
    out_path: Path,
    cfg: VitPoseEuro2DConfig | None = None,
    segment: OfflineSessionSegment | None = None,
    max_frames: int = 0,
    viterbi_out_path: Path | None = None,
    **kwargs,
) -> tuple[VitPoseEuro2DStats, dict]:
    """兼容旧入口。"""
    _ = kwargs
    return run_filter_pose2d_vitpose(
        pose_path=pose_path,
        out_path=out_path,
        viterbi_out_path=viterbi_out_path,
        cfg=cfg,
        segment=segment,
        max_frames=max_frames,
    )
