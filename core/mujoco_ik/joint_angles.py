"""MuJoCo IK 关节角读取（供 3D preview 叠加）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

KNEE_JOINT_NAMES: tuple[str, str] = ("left_knee", "right_knee")
# wb133: left_knee / right_knee；官方 humanoid: knee_left / knee_right
KNEE_JOINT_ALIASES: dict[str, tuple[str, ...]] = {
    "left_knee": ("left_knee", "knee_left"),
    "right_knee": ("right_knee", "knee_right"),
}


@dataclass(frozen=True)
class KneeFlexAngles:
    """膝 hinge 角（度）：0=伸直，正=屈曲，负=过伸。"""

    left_deg: float
    right_deg: float

    def overlay_line(self) -> str:
        return (
            f"L knee {self.left_deg:+.1f} deg  "
            f"R knee {self.right_deg:+.1f} deg  "
            f"(+flex -ext)"
        )

    def status_suffix(self) -> str:
        return f"Lk={self.left_deg:.0f} Rk={self.right_deg:.0f}"


def _resolve_joint_qposadr(model, mujoco_mod, names: tuple[str, ...]) -> int | None:
    for name in names:
        jid = mujoco_mod.mj_name2id(model, mujoco_mod.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            return int(model.jnt_qposadr[jid])
    return None


def knee_qpos_indices(model, mujoco_mod) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in KNEE_JOINT_NAMES:
        aliases = KNEE_JOINT_ALIASES.get(key, (key,))
        adr = _resolve_joint_qposadr(model, mujoco_mod, aliases)
        if adr is None:
            raise ValueError(f"MJCF 缺少 knee joint: {key} (tried {aliases})")
        out[key] = adr
    return out


def knee_flex_angles_deg(
    qpos: np.ndarray,
    *,
    knee_qpos_idx: dict[str, int],
) -> KneeFlexAngles:
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    return KneeFlexAngles(
        left_deg=float(np.degrees(q[knee_qpos_idx["left_knee"]])),
        right_deg=float(np.degrees(q[knee_qpos_idx["right_knee"]])),
    )
