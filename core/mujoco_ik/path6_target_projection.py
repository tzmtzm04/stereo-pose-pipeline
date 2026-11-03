"""Path6 MSK：IK 前目标投影（骨长 FABRIK；角限在 IK/MJCF）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.mujoco_ik.target_projection import _fabrik3, _get, _limit_shift, _set


@dataclass(frozen=True)
class Path6TargetProjectionConfig:
    """MSK 骨段长度近似（Rajagopal / wb133 量级）。"""

    upper_arm_m: float = 0.26
    forearm_m: float = 0.24
    thigh_m: float = 0.40
    shin_m: float = 0.40
    max_point_shift_m: float = 0.25


_LEG_CHAINS: tuple[tuple[str, str, str, str, str], ...] = (
    ("LHip_marker", "LKnee_marker", "LAnkle_marker", "thigh_m", "shin_m"),
    ("RHip_marker", "RKnee_marker", "RAnkle_marker", "thigh_m", "shin_m"),
)

_ARM_CHAINS: tuple[tuple[str, str, str, str, str], ...] = (
    ("LShoulder_marker", "LElbow_marker", "LWrist_marker", "upper_arm_m", "forearm_m"),
    ("RShoulder_marker", "RElbow_marker", "RWrist_marker", "upper_arm_m", "forearm_m"),
)


class Path6TargetProjector:
    """将 path6 marker site 目标投影到固定骨长。"""

    def __init__(self, cfg: Path6TargetProjectionConfig | None = None) -> None:
        self.cfg = cfg or Path6TargetProjectionConfig()

    def project(
        self,
        targets: dict[str, np.ndarray | None],
        *,
        qpos_hint: np.ndarray | None = None,
        targets_prev: dict[str, np.ndarray | None] = None,
    ) -> dict[str, np.ndarray | None]:
        del qpos_hint, targets_prev
        out: dict[str, np.ndarray | None] = {}
        for name, val in targets.items():
            if val is None:
                out[name] = None
            else:
                out[name] = np.asarray(val, dtype=np.float64).reshape(3).copy()

        self._project_chains(out, _LEG_CHAINS)
        self._project_chains(out, _ARM_CHAINS)
        return out

    def _apply_point(
        self,
        targets: dict[str, np.ndarray | None],
        name: str,
        orig: np.ndarray,
        moved: np.ndarray,
    ) -> None:
        _set(
            targets,
            name,
            _limit_shift(orig, moved, self.cfg.max_point_shift_m),
        )

    def _project_chains(
        self,
        targets: dict[str, np.ndarray | None],
        chains: tuple[tuple[str, str, str, str, str], ...],
    ) -> None:
        cfg = self.cfg
        for root_n, mid_n, tip_n, len_ab, len_bc in chains:
            a = _get(targets, root_n)
            b = _get(targets, mid_n)
            c = _get(targets, tip_n)
            if a is None or b is None or c is None:
                continue
            lab = float(getattr(cfg, len_ab))
            lbc = float(getattr(cfg, len_bc))
            _a, b2, c2 = _fabrik3(a, b, c, lab, lbc)
            b_orig, c_orig = b.copy(), c.copy()
            self._apply_point(targets, mid_n, b_orig, b2)
            self._apply_point(targets, tip_n, c_orig, c2)
