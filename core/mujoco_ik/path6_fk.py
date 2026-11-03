"""Path 6 MSK MJCF：qpos → marker site FK。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.mujoco_ik.path6_skeleton_layout import (
    PATH6_KNEE_JOINT_NAMES,
    PATH6_PREVIEW_SITE_NAMES,
)


@dataclass(frozen=True)
class Path6KneeFlexAngles:
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


class Path6MujocoFK:
    """MSK MJCF FK：qpos → 13 marker site xyz。"""

    def __init__(
        self,
        model_path: Path | str,
        *,
        site_names: tuple[str, ...] = PATH6_PREVIEW_SITE_NAMES,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self.site_names = site_names
        self._site_ids: dict[str, int] = {}
        for name in site_names:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, name)
            if sid < 0:
                raise ValueError(f"MJCF 缺少 site: {name}")
            self._site_ids[name] = int(sid)

        self._knee_qpos_idx: dict[str, int] = {}
        for side, jname in zip(("left", "right"), PATH6_KNEE_JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                continue
            self._knee_qpos_idx[side] = int(self.model.jnt_qposadr[jid])

    def knee_flex_angles(self, qpos: np.ndarray) -> Path6KneeFlexAngles:
        q = np.asarray(qpos, dtype=np.float64).reshape(-1)
        left = float(np.degrees(q[self._knee_qpos_idx["left"]])) if "left" in self._knee_qpos_idx else float("nan")
        right = float(np.degrees(q[self._knee_qpos_idx["right"]])) if "right" in self._knee_qpos_idx else float("nan")
        return Path6KneeFlexAngles(left_deg=left, right_deg=right)

    def site_kpts3d(self, qpos: np.ndarray) -> np.ndarray:
        """(13,4) marker site 位置，conf=1。"""
        mujoco = self._mujoco
        self.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        n = len(self.site_names)
        kpts = np.zeros((n, 4), dtype=np.float64)
        for i, name in enumerate(self.site_names):
            sid = self._site_ids[name]
            kpts[i, :3] = self.data.site_xpos[sid]
            kpts[i, 3] = 1.0
        return kpts
