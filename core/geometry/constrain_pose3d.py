"""Stage 4：3D 关键点硬投影（骨长 FABRIK + 躯干避让；角限在 IK/MJCF）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.mujoco_ik.layout import apply_ik_targets_to_kpts3d, wb133_to_ik_targets
from core.mujoco_ik.target_projection import (
    TargetProjectionConfig,
    TargetConstraintProjector,
    max_angle_violation_deg,
    max_bone_violation_m,
)
from core.offline.session_segment import OfflineSessionSegment


@dataclass(frozen=True)
class ConstrainPose3DConfig:
    """Stage 4 硬投影（无观测 blend、无位移上限）。"""

    min_conf: float = 0.0
    projection: TargetProjectionConfig = TargetProjectionConfig(
        hard_projection=True,
        blend_prev=0.0,
        projection_priority="bone_first",
        enable_joint_limits=False,
    )


@dataclass
class ConstrainPose3DStats:
    frames_in_raw: int = 0
    frames_in: int = 0
    frames_out: int = 0
    frames_skip: int = 0
    segment: dict | None = None
    mean_shift_m: float = 0.0
    max_shift_m: float = 0.0
    bone_viol_before_m: float = 0.0
    bone_viol_after_m: float = 0.0
    angle_viol_before_deg: float = 0.0
    angle_viol_after_deg: float = 0.0


def _read_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _dedupe_rows_by_capture_seq(rows: list[dict]) -> list[dict]:
    by_seq: dict[int, dict] = {}
    for row in rows:
        by_seq[int(row.get("capture_seq", -1))] = row
    return [by_seq[k] for k in sorted(by_seq)]


def _apply_segment_rows(
    rows: list[dict],
    segment: OfflineSessionSegment | None,
) -> tuple[list[dict], OfflineSessionSegment | None]:
    if segment is None:
        return rows, None
    sl = segment.slice(len(rows))
    return rows[sl], segment


def _max_bone_violation(
    targets: dict[str, np.ndarray | None],
    cfg: TargetProjectionConfig,
) -> float:
    return max_bone_violation_m(targets, cfg)


def _max_angle_violation(
    targets: dict[str, np.ndarray | None],
    cfg: TargetProjectionConfig,
) -> float:
    return max_angle_violation_deg(targets, cfg)


def run_session_constrain3d(
    *,
    pose3d_path: Path,
    out_path: Path,
    cfg: ConstrainPose3DConfig | None = None,
    segment: OfflineSessionSegment | None = None,
    max_frames: int = 0,
) -> ConstrainPose3DStats:
    cfg = cfg or ConstrainPose3DConfig()
    projector = TargetConstraintProjector(cfg=cfg.projection)

    raw_rows = _read_jsonl_rows(pose3d_path)
    rows = _dedupe_rows_by_capture_seq(raw_rows)
    if max_frames > 0:
        rows = rows[: int(max_frames)]
    rows, seg_used = _apply_segment_rows(rows, segment)

    stats = ConstrainPose3DStats(
        frames_in_raw=len(raw_rows),
        frames_in=len(rows),
        segment=seg_used.to_dict() if seg_used is not None else None,
    )

    shifts: list[float] = []
    viol_before: list[float] = []
    viol_after: list[float] = []
    angle_before: list[float] = []
    angle_after: list[float] = []

    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as out_f:
        for row in rows:
            kpts_raw = row.get("kpts3d")
            if not kpts_raw:
                stats.frames_skip += 1
                continue
            kpts = np.asarray(kpts_raw, dtype=np.float64).reshape(-1, 4)
            if not np.any(np.isfinite(kpts[:, :3])):
                stats.frames_skip += 1
                continue

            targets = wb133_to_ik_targets(kpts, min_conf=float(cfg.min_conf))
            viol_before.append(_max_bone_violation(targets, cfg.projection))
            angle_before.append(_max_angle_violation(targets, cfg.projection))

            projected = projector.project(targets)
            viol_after.append(_max_bone_violation(projected, cfg.projection))
            angle_after.append(_max_angle_violation(projected, cfg.projection))

            kpts_new = apply_ik_targets_to_kpts3d(kpts, projected)
            delta = np.linalg.norm(
                np.nan_to_num(kpts_new[:, :3]) - np.nan_to_num(kpts[:, :3]),
                axis=1,
            )
            mask = np.isfinite(kpts[:, :3]).all(axis=1) & np.isfinite(kpts_new[:, :3]).all(
                axis=1
            )
            if np.any(mask):
                shifts.append(float(np.max(delta[mask])))

            rec = dict(row)
            rec["kpts3d"] = kpts_new.tolist()
            rec["tri_ok"] = bool(np.any(np.isfinite(kpts_new[:, :3])))
            rec["constrain3d"] = True
            rec["constrain3d_hard"] = bool(cfg.projection.hard_projection)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            stats.frames_out += 1

    if shifts:
        stats.mean_shift_m = float(np.mean(shifts))
        stats.max_shift_m = float(np.max(shifts))
    if viol_before:
        stats.bone_viol_before_m = float(np.mean(viol_before))
    if viol_after:
        stats.bone_viol_after_m = float(np.mean(viol_after))
    if angle_before:
        stats.angle_viol_before_deg = float(np.mean(angle_before))
    if angle_after:
        stats.angle_viol_after_deg = float(np.mean(angle_after))
    return stats
