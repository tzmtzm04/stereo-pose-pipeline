"""G1 IK 前目标投影：复用 Stage 4 / TargetConstraintProjector FABRIK 骨长。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.mujoco_ik.g1_fk import G1BodyFK, G1_BODY_FRAMES
from core.mujoco_ik.target_projection import TargetConstraintProjector, TargetProjectionConfig

# G1 task key → wb133/COCO 投影名（与 target_projection._LIMB_CHAINS 一致）
G1_KEY_TO_PROJ: dict[str, str] = {
    "left_shoulder_yaw_link": "left_shoulder",
    "right_shoulder_yaw_link": "right_shoulder",
    "left_elbow_link": "left_elbow",
    "right_elbow_link": "right_elbow",
    "left_palm": "left_wrist",
    "right_palm": "right_wrist",
    "left_hip_yaw_link": "left_hip",
    "right_hip_yaw_link": "right_hip",
    "left_knee_link": "left_knee",
    "right_knee_link": "right_knee",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
}
PROJ_TO_G1_KEY: dict[str, str] = {v: k for k, v in G1_KEY_TO_PROJ.items()}

_IDX = {name: i for i, (name, _) in enumerate(G1_BODY_FRAMES)}


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)))


def _hard_projection_cfg(**overrides: float | str | bool) -> TargetProjectionConfig:
    base = dict(
        hard_projection=True,
        projection_priority="bone_first",
        enable_joint_limits=False,
    )
    base.update(overrides)
    return TargetProjectionConfig(**base)


def projection_config_from_constrain3d_summary(
    session_dir: Path | str,
    *,
    summary_path: Path | None = None,
) -> tuple[TargetProjectionConfig, str]:
    """从 session 的 constrained_3d_summary.json 读取 Stage 4 骨长（与 IK 前约束一致）。"""
    session = Path(session_dir).resolve()
    path = summary_path or (session / "constrained_3d_summary.json")
    if not path.is_file():
        return _hard_projection_cfg(), "default"

    data = json.loads(path.read_text(encoding="utf-8"))
    c3 = data.get("constrain3d") or {}
    bl = c3.get("bone_lengths_m") or {}
    cfg = _hard_projection_cfg(
        upper_arm_m=float(bl.get("upper_arm", 0.26)),
        forearm_m=float(bl.get("forearm", 0.24)),
        thigh_m=float(bl.get("thigh", 0.40)),
        shin_m=float(bl.get("shin", 0.40)),
        torso_capsule_radius_m=float(c3.get("torso_capsule_radius_m", 0.10)),
        torso_clearance_m=float(c3.get("torso_clearance_m", 0.05)),
        projection_priority=str(c3.get("projection_priority", "bone_first")),
    )
    return cfg, str(path)


def measure_g1_bone_lengths(model_path: Path | str) -> TargetProjectionConfig:
    """stand keyframe 下测 G1 MJCF 上臂/前臂/大腿/小腿长度（可选，非默认）。"""
    import mujoco

    p = Path(model_path).resolve()
    model = mujoco.MjModel.from_xml_path(str(p))
    data = mujoco.MjData(model)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)

    fk = G1BodyFK(p)
    fk.data.qpos[:] = data.qpos
    fk._mujoco.mj_forward(fk.model, fk.data)
    k = fk.kpts3d(data.qpos)

    return _hard_projection_cfg(
        upper_arm_m=_dist(k[_IDX["left_shoulder_yaw_link"]], k[_IDX["left_elbow_link"]]),
        forearm_m=_dist(k[_IDX["left_elbow_link"]], k[_IDX["left_palm"]]),
        thigh_m=_dist(k[_IDX["left_hip_yaw_link"]], k[_IDX["left_knee_link"]]),
        shin_m=_dist(k[_IDX["left_knee_link"]], k[_IDX["left_foot"]]),
    )


@dataclass
class G1TargetProjector:
    """观测 3D → FABRIK 骨长投影 → 再进 mink FrameTask。"""

    cfg: TargetProjectionConfig

    def __post_init__(self) -> None:
        self._inner = TargetConstraintProjector(cfg=self.cfg)

    @property
    def bone_lengths(self) -> dict[str, float]:
        c = self.cfg
        return {
            "upper_arm_m": float(c.upper_arm_m),
            "forearm_m": float(c.forearm_m),
            "thigh_m": float(c.thigh_m),
            "shin_m": float(c.shin_m),
        }

    def _to_proj(self, targets: dict[str, np.ndarray | None]) -> dict[str, np.ndarray | None]:
        out: dict[str, np.ndarray | None] = {name: None for name in PROJ_TO_G1_KEY}
        for g1_key, proj_name in G1_KEY_TO_PROJ.items():
            p = targets.get(g1_key)
            if p is not None:
                out[proj_name] = np.asarray(p, dtype=np.float64).reshape(3).copy()
        pelvis = targets.get("pelvis")
        if pelvis is not None:
            out["pelvis"] = np.asarray(pelvis, dtype=np.float64).reshape(3).copy()
        return out

    def _from_proj(
        self,
        g1_in: dict[str, np.ndarray | None],
        proj_out: dict[str, np.ndarray | None],
    ) -> dict[str, np.ndarray | None]:
        out = dict(g1_in)
        for proj_name, g1_key in PROJ_TO_G1_KEY.items():
            p = proj_out.get(proj_name)
            if p is not None:
                out[g1_key] = np.asarray(p, dtype=np.float64).reshape(3).copy()
        return out

    def project(
        self,
        targets: dict[str, np.ndarray | None],
        *,
        qpos_hint: np.ndarray | None = None,
        targets_prev: dict[str, np.ndarray | None] | None = None,
    ) -> dict[str, np.ndarray | None]:
        del qpos_hint
        proj_in = self._to_proj(targets)
        proj_prev = self._to_proj(targets_prev) if targets_prev else None
        proj_out = self._inner.project(proj_in, targets_prev=proj_prev)
        return self._from_proj(targets, proj_out)
