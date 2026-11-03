"""full_session_pose3d.jsonl → MuJoCo IK 批处理（DLS / mink）。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from typing import Callable

from core.mujoco_ik.factory import (
    IKBackend,
    default_min_targets,
    ik_method_label,
    make_ik_solver,
    normalize_ik_backend,
)
from core.mujoco_ik.jacobian_ik import JacobianIKConfig
from core.mujoco_ik.mink_ik import MinkIKConfig, MinkIKSolver
from core.mujoco_ik.extremity_ik import ExtremityIKConfig, ExtremityIKSolver
from core.mujoco_ik.layout import wb133_to_ik_targets
from core.mujoco_ik.root_facing import (
    RootFacingConfig,
    RootFacingTracker,
    model_root_on_pelvis,
)
from core.mujoco_ik.target_projection import TargetConstraintProjector


@dataclass
class SessionIKStats:
    frames_in_raw: int = 0
    frames_in: int = 0
    frames_ok: int = 0
    frames_skip: int = 0
    mean_residual: float = 0.0


def _read_jsonl_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _dedupe_pose_rows_by_capture_seq(rows: list[dict]) -> list[dict]:
    """同一 capture_seq 保留最后一条（与 offline_preview / load_mujoco_ik_sequence 一致）。"""
    by_seq: dict[int, dict] = {}
    for row in rows:
        by_seq[int(row.get("capture_seq", -1))] = row
    return [by_seq[k] for k in sorted(by_seq)]


def _read_capture_fps(pose_path: Path, default: float = 60.0) -> float:
    meta_path = pose_path.parent / "full_session_meta.json"
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        fps = float(meta.get("fps", default) or default)
        if fps > 0.0:
            return fps
    return default


def run_session_mujoco_ik(
    pose_path: Path,
    *,
    model_path: Path,
    out_jsonl: Path,
    min_conf: float = 0.35,
    tri_ok_only: bool = True,
    backend: IKBackend | str = "dls",
    ik_cfg: JacobianIKConfig | MinkIKConfig | None = None,
    dls_cfg: JacobianIKConfig | None = None,
    mink_cfg: MinkIKConfig | None = None,
    capture_fps: float | None = None,
    extremity_cfg: ExtremityIKConfig | None = None,
    to_ik_targets: Callable[..., dict[str, np.ndarray | None]] | None = None,
    solver: object | None = None,
    target_projector: object | None = None,
    root_facing: bool | None = None,
) -> SessionIKStats:
    backend_key = normalize_ik_backend(str(backend))
    if backend_key == "mink":
        mink_cfg = mink_cfg or (
            ik_cfg if isinstance(ik_cfg, MinkIKConfig) else MinkIKConfig()
        )
        dls_cfg = dls_cfg or JacobianIKConfig()
    else:
        dls_cfg = dls_cfg or (
            ik_cfg if isinstance(ik_cfg, JacobianIKConfig) else JacobianIKConfig()
        )
        mink_cfg = mink_cfg or MinkIKConfig()

    solver = solver or make_ik_solver(
        model_path,
        backend=backend_key,
        dls_cfg=dls_cfg,
        mink_cfg=mink_cfg,
    )
    target_mapper = to_ik_targets or wb133_to_ik_targets
    min_targets = default_min_targets(
        backend_key, dls_cfg=dls_cfg, mink_cfg=mink_cfg
    )
    ik_method = ik_method_label(backend_key)

    target_projector_obj: object | None = target_projector
    if target_projector_obj is None and (
        backend_key == "mink"
        and mink_cfg is not None
        and mink_cfg.project_targets
    ):
        target_projector_obj = TargetConstraintProjector(model_path)

    ext_cfg = extremity_cfg if extremity_cfg is not None else ExtremityIKConfig()
    extremity_solver: ExtremityIKSolver | None = None
    if ext_cfg.enabled:
        extremity_solver = ExtremityIKSolver(model_path, ext_cfg)

    use_root_facing = (
        bool(mink_cfg.root_facing)
        if backend_key == "mink"
        else bool(dls_cfg.root_facing)
    )
    if root_facing is not None:
        use_root_facing = bool(root_facing)
    root_tracker: RootFacingTracker | None = None
    if use_root_facing:
        import mujoco

        mj_model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        lock_trans = model_root_on_pelvis(mj_model, mujoco)
        hard_lock = (
            bool(mink_cfg.root_facing_hard_lock)
            if backend_key == "mink"
            else bool(dls_cfg.root_facing_hard_lock)
        )
        root_tracker = RootFacingTracker(
            RootFacingConfig(
                enabled=True,
                mode="yaw_correct",
                lock_translation=lock_trans,
                hard_lock_rotation=hard_lock,
            )
        )

    raw_rows = _read_jsonl_rows(pose_path)
    rows = _dedupe_pose_rows_by_capture_seq(raw_rows)
    stats = SessionIKStats(frames_in_raw=len(raw_rows), frames_in=len(rows))
    residuals: list[float] = []

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    qpos_prev: np.ndarray | None = None
    targets_prev: dict[str, np.ndarray | None] | None = None

    with out_jsonl.open("w", encoding="utf-8") as out_f:
        for row in rows:
            if tri_ok_only and not row.get("tri_ok", True):
                stats.frames_skip += 1
                continue
            kpts = row.get("kpts3d")
            if not kpts:
                stats.frames_skip += 1
                continue

            targets = target_mapper(kpts, min_conf=min_conf)
            if target_projector_obj is not None:
                targets = target_projector_obj.project(
                    targets,
                    qpos_hint=qpos_prev,
                    targets_prev=targets_prev,
                )
            root_lock = None
            if root_tracker is not None:
                root_lock = root_tracker.lock_from_targets(targets, qpos_prev)
            if backend_key == "mink" and isinstance(solver, MinkIKSolver):
                qpos, residual, n_tgt = solver.track_capture(
                    targets,
                    targets_prev=targets_prev,
                    qpos_anchor=qpos_prev,
                    reset_session=targets_prev is None and qpos_prev is None,
                    root_lock=root_lock,
                )
            else:
                qpos, residual, n_tgt = solver.solve(
                    targets,
                    qpos_init=qpos_prev,
                    root_lock=root_lock,
                )

            if not np.isfinite(residual) or n_tgt < min_targets:
                stats.frames_skip += 1
                continue

            if extremity_solver is not None:
                qpos = extremity_solver.apply(qpos, np.asarray(kpts), min_conf=min_conf)

            if (
                root_tracker is not None
                and root_lock is not None
                and not root_lock.hard_lock_rotation
            ):
                corrected = root_tracker.correct_qpos_root(qpos, targets)
                if corrected is not None:
                    qpos = corrected

            qpos_prev = qpos.copy()
            targets_prev = targets
            stats.frames_ok += 1
            if np.isfinite(residual):
                residuals.append(residual)

            rec = {
                "capture_seq": int(row.get("capture_seq", -1)),
                "infer_seq": int(row.get("infer_seq", -1)),
                "tri_ok": bool(row.get("tri_ok", True)),
                "n_targets": int(n_tgt),
                "residual": float(residual),
                "ik_method": ik_method,
                "joint_names": solver.joint_names,
                "qpos": qpos.tolist(),
            }
            if root_lock is not None:
                rec["root_facing"] = True
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if residuals:
        stats.mean_residual = float(np.mean(residuals))
    return stats
