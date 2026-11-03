"""mujoco_ik.jsonl → qpos FK → site 3D 点。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.mujoco_ik.jacobian_ik import JacobianIKSolver
from core.mujoco_ik.joint_angles import KneeFlexAngles, knee_flex_angles_deg, knee_qpos_indices
from core.mujoco_ik.skeleton_layout import MUJOCO_IK_PREVIEW_SITE_NAMES


@dataclass(frozen=True)
class MujocoIKFrame:
    capture_seq: int
    infer_seq: int
    tri_ok: bool
    n_targets: int
    residual: float
    qpos: np.ndarray


def load_mujoco_ik_sequence(
    path: Path,
    *,
    start_capture_seq: int | None = None,
    end_capture_seq: int | None = None,
    stride: int = 1,
) -> list[MujocoIKFrame]:
    """读 mujoco_ik.jsonl；按 capture_seq 去重（保留最后一条）。"""
    by_seq: dict[int, MujocoIKFrame] = {}
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qpos = np.asarray(row["qpos"], dtype=np.float64)
            frame = MujocoIKFrame(
                capture_seq=int(row.get("capture_seq", -1)),
                infer_seq=int(row.get("infer_seq", -1)),
                tri_ok=bool(row.get("tri_ok", True)),
                n_targets=int(row.get("n_targets", 0)),
                residual=float(row.get("residual", float("nan"))),
                qpos=qpos,
            )
            by_seq[frame.capture_seq] = frame

    rows = sorted(by_seq.values(), key=lambda r: r.capture_seq)
    if start_capture_seq is not None:
        rows = [r for r in rows if r.capture_seq >= int(start_capture_seq)]
    if end_capture_seq is not None:
        rows = [r for r in rows if r.capture_seq <= int(end_capture_seq)]
    if int(stride) > 1:
        rows = rows[:: int(stride)]
    return rows


class MujocoFK:
    """MJCF 模型 FK：qpos → site xyz。"""

    def __init__(self, model_path: Path | str) -> None:
        self._solver = JacobianIKSolver(model_path)
        mujoco = self._solver._mujoco
        model = self._solver.model
        self._preview_site_ids: dict[str, int] = {}
        for name in MUJOCO_IK_PREVIEW_SITE_NAMES:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"MJCF 缺少 preview site: {name}")
            self._preview_site_ids[name] = int(sid)
        self._knee_qpos_idx = knee_qpos_indices(
            self._solver.model,
            self._solver._mujoco,
        )

    def knee_flex_angles(self, qpos: np.ndarray) -> KneeFlexAngles:
        """膝屈伸角（度）：MJCF hinge，0=伸直，正=屈曲。"""
        return knee_flex_angles_deg(
            qpos,
            knee_qpos_idx=self._knee_qpos_idx,
        )

    def site_kpts3d(self, qpos: np.ndarray) -> np.ndarray:
        """(N,4) preview site 位置（含腕踝 FK），conf 固定 1.0。"""
        mujoco = self._solver._mujoco
        self._solver.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        mujoco.mj_forward(self._solver.model, self._solver.data)
        n = len(MUJOCO_IK_PREVIEW_SITE_NAMES)
        kpts = np.zeros((n, 4), dtype=np.float64)
        data = self._solver.data
        for i, name in enumerate(MUJOCO_IK_PREVIEW_SITE_NAMES):
            sid = self._preview_site_ids[name]
            kpts[i, :3] = data.site_xpos[sid]
            kpts[i, 3] = 1.0
        return kpts
