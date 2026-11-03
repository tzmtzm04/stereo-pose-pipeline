"""Unitree G1 + mink 官方 humanoid_g1.py task 布局（session IK）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.mujoco_ik.g1_layout import (
    G1_ELBOW_BODIES,
    G1_EXTREMITY_SITES,
    G1_FOOT_SITES,
    G1_HAND_SITES,
    G1_HIP_BODIES,
    G1_KNEE_BODIES,
    G1_LIMB_FRAME_SPECS,
    G1_SHOULDER_BODIES,
    g1_limb_task_keys,
)
from core.mujoco_ik.qpos_limits import clip_qpos_to_limits
from core.mujoco_ik.root_facing import estimate_torso_frame


def _keyframe_qpos(model, mujoco_mod, name: str) -> np.ndarray | None:
    kid = mujoco_mod.mj_name2id(model, mujoco_mod.mjtObj.mjOBJ_KEY, name)
    if kid < 0:
        return None
    data = mujoco_mod.MjData(model)
    mujoco_mod.mj_resetDataKeyframe(model, data, kid)
    return data.qpos.copy()


@dataclass
class G1MinkIKConfig:
    max_steps: int = 50
    tol: float = 1e-3
    min_targets: int = 4
    step_dt: float = 0.005
    damping: float = 0.1
    solver: str = "daqp"
    posture_cost: float = 0.1
    com_cost: float = 10.0
    pelvis_orientation_cost: float = 1.0
    torso_orientation_cost: float = 1.0
    foot_position_cost: float = 10.0
    foot_orientation_cost: float = 0.0
    hand_position_cost: float = 2.0
    hand_orientation_cost: float = 0.0
    elbow_position_cost: float = 5.0
    knee_position_cost: float = 8.0
    shoulder_position_cost: float = 6.0
    hip_position_cost: float = 7.0
    official_limb_tasks_only: bool = False
    full_body_limb_tasks: bool = False
    lm_damping: float = 1.0
    keyframe_name: str = "stand"
    stall_steps: int = 5


class G1MinkIKSolver:
    """G1 mink：pelvis/torso ori + Posture + COM + 腕踝肘膝 FrameTask。"""

    def __init__(
        self,
        model_path: Path | str,
        cfg: G1MinkIKConfig | None = None,
    ) -> None:
        try:
            import mink
        except ImportError as exc:
            raise RuntimeError("未安装 mink。请: pip install mink qpsolvers daqp") from exc

        import mujoco

        self._mink = mink
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.cfg = cfg or G1MinkIKConfig()
        self._limb_task_keys = g1_limb_task_keys(
            official_only=bool(self.cfg.official_limb_tasks_only),
            full_body=bool(self.cfg.full_body_limb_tasks),
        )

        self._configuration = mink.Configuration(self.model)
        q0 = _keyframe_qpos(self.model, mujoco, self.cfg.keyframe_name)
        if q0 is None:
            q0 = self.model.qpos0.copy()
        self._default_qpos = q0.copy()
        self._configuration.update(q=self._default_qpos)

        self._pelvis_ori_task = mink.FrameTask(
            "pelvis",
            "body",
            position_cost=0.0,
            orientation_cost=float(self.cfg.pelvis_orientation_cost),
            lm_damping=float(self.cfg.lm_damping),
        )
        self._torso_ori_task = mink.FrameTask(
            "torso_link",
            "body",
            position_cost=0.0,
            orientation_cost=float(self.cfg.torso_orientation_cost),
            lm_damping=float(self.cfg.lm_damping),
        )
        self._posture_task = mink.PostureTask(
            self.model, cost=float(self.cfg.posture_cost)
        )
        self._com_task = mink.ComTask(cost=float(self.cfg.com_cost))

        cfg = self.cfg
        cost_by_key: dict[str, float] = {}
        for foot in G1_FOOT_SITES:
            cost_by_key[foot] = float(cfg.foot_position_cost)
        for hand in G1_HAND_SITES:
            cost_by_key[hand] = float(cfg.hand_position_cost)
        for elbow in G1_ELBOW_BODIES:
            cost_by_key[elbow] = float(cfg.elbow_position_cost)
        for knee in G1_KNEE_BODIES:
            cost_by_key[knee] = float(cfg.knee_position_cost)
        for shoulder in G1_SHOULDER_BODIES:
            cost_by_key[shoulder] = float(cfg.shoulder_position_cost)
        for hip in G1_HIP_BODIES:
            cost_by_key[hip] = float(cfg.hip_position_cost)

        ori_by_key: dict[str, float] = {}
        for site in G1_EXTREMITY_SITES:
            ori_by_key[site] = (
                float(cfg.foot_orientation_cost)
                if site in G1_FOOT_SITES
                else float(cfg.hand_orientation_cost)
            )
        for body in G1_ELBOW_BODIES + G1_KNEE_BODIES + G1_SHOULDER_BODIES + G1_HIP_BODIES:
            ori_by_key[body] = 0.0

        self._limb_tasks: dict[str, Any] = {}
        for key in self._limb_task_keys:
            frame_name, frame_type = G1_LIMB_FRAME_SPECS[key]
            self._limb_tasks[key] = mink.FrameTask(
                frame_name,
                frame_type,
                position_cost=float(cost_by_key[key]),
                orientation_cost=float(ori_by_key[key]),
                lm_damping=float(cfg.lm_damping),
            )

        self._limits = [mink.ConfigurationLimit(self.model)]
        self._init_task_targets_from_configuration()

    @property
    def joint_names(self) -> list[str]:
        names: list[str] = []
        for j in range(self.model.njnt):
            n = self._mujoco.mj_id2name(
                self.model, self._mujoco.mjtObj.mjOBJ_JOINT, j
            )
            names.append(n or f"joint_{j}")
        return names

    def reset(self) -> None:
        self._configuration.update(q=self._default_qpos.copy())
        self._init_task_targets_from_configuration()

    def _init_task_targets_from_configuration(self) -> None:
        self._posture_task.set_target_from_configuration(self._configuration)
        self._pelvis_ori_task.set_target_from_configuration(self._configuration)
        self._torso_ori_task.set_target_from_configuration(self._configuration)
        self._com_task.set_target(self._configuration.data.subtree_com[1].copy())

    def _se3_from_torso_frame(self, frame: tuple[np.ndarray, np.ndarray, np.ndarray]) -> Any:
        mink = self._mink
        right, up, forward = frame
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = np.column_stack(
            [
                np.asarray(right, dtype=np.float64),
                np.asarray(up, dtype=np.float64),
                np.asarray(forward, dtype=np.float64),
            ]
        )
        pelvis_tf = self._configuration.get_transform_frame_to_world("pelvis", "body")
        mat[:3, 3] = pelvis_tf.translation()
        return mink.SE3.from_matrix(mat)

    def _update_orientation_tasks(self, obs: dict[str, np.ndarray | None]) -> None:
        frame = estimate_torso_frame(obs)
        if frame is None:
            return
        se3 = self._se3_from_torso_frame(frame)
        self._pelvis_ori_task.set_target(se3)
        torso_tf = self._configuration.get_transform_frame_to_world("torso_link", "body")
        mat = se3.as_matrix().copy()
        mat[:3, 3] = torso_tf.translation()
        self._torso_ori_task.set_target(self._mink.SE3.from_matrix(mat))

    def _active_limb_targets(
        self, targets: dict[str, np.ndarray | None]
    ) -> dict[str, np.ndarray]:
        active: dict[str, np.ndarray] = {}
        for name in self._limb_task_keys:
            p = targets.get(name)
            if p is None:
                continue
            arr = np.asarray(p, dtype=np.float64).reshape(3)
            if np.all(np.isfinite(arr)) and np.linalg.norm(arr) > 1e-4:
                active[name] = arr
        return active

    def _set_limb_targets(self, active: dict[str, np.ndarray]) -> None:
        mink = self._mink
        for key, pos in active.items():
            task = self._limb_tasks[key]
            frame_name, frame_type = G1_LIMB_FRAME_SPECS[key]
            cur = self._configuration.get_transform_frame_to_world(
                frame_name, frame_type
            )
            mat = cur.as_matrix().copy()
            mat[:3, 3] = pos
            task.set_target(mink.SE3.from_matrix(mat))

    def _frame_position(self, key: str) -> np.ndarray:
        frame_name, frame_type = G1_LIMB_FRAME_SPECS[key]
        data = self._configuration.data
        if frame_type == "site":
            sid = self._mujoco.mj_name2id(
                self.model, self._mujoco.mjtObj.mjOBJ_SITE, frame_name
            )
            return np.asarray(data.site_xpos[sid], dtype=np.float64)
        bid = self._mujoco.mj_name2id(
            self.model, self._mujoco.mjtObj.mjOBJ_BODY, frame_name
        )
        return np.asarray(data.xpos[bid], dtype=np.float64)

    def _residual(self, active: dict[str, np.ndarray]) -> float:
        self._mujoco.mj_forward(self.model, self._configuration.data)
        errs = [tgt - self._frame_position(name) for name, tgt in active.items()]
        if not errs:
            return float("inf")
        return float(np.linalg.norm(np.concatenate(errs)))

    def solve(
        self,
        targets: dict[str, np.ndarray | None],
        *,
        qpos_init: np.ndarray | None = None,
        root_lock: object | None = None,
    ) -> tuple[np.ndarray, float, int]:
        del root_lock
        cfg = self.cfg
        active = self._active_limb_targets(targets)
        if len(active) < cfg.min_targets:
            q = (
                np.asarray(qpos_init, dtype=np.float64).copy()
                if qpos_init is not None
                else self._configuration.q.copy()
            )
            return q, float("inf"), len(active)

        if qpos_init is not None:
            self._configuration.update(q=np.asarray(qpos_init, dtype=np.float64).copy())
        self._posture_task.set_target(
            np.asarray(qpos_init, dtype=np.float64).copy()
            if qpos_init is not None
            else self._configuration.q.copy()
        )

        self._update_orientation_tasks(targets)
        com = targets.get("com")
        if com is not None:
            self._com_task.set_target(np.asarray(com, dtype=np.float64).reshape(3))
        self._set_limb_targets(active)

        tasks: list[Any] = [
            self._pelvis_ori_task,
            self._torso_ori_task,
            self._posture_task,
            self._com_task,
            *self._limb_tasks.values(),
        ]

        step_dt = float(cfg.step_dt)
        last_err = self._residual(active)
        stall = 0
        prev_err = last_err

        for _ in range(cfg.max_steps):
            try:
                vel = self._mink.solve_ik(
                    self._configuration,
                    tasks,
                    dt=step_dt,
                    solver=cfg.solver,
                    damping=float(cfg.damping),
                    limits=self._limits,
                )
            except self._mink.NoSolutionFound:
                break

            self._configuration.integrate_inplace(vel, step_dt)
            last_err = self._residual(active)
            if last_err < cfg.tol:
                break
            if last_err >= prev_err * 0.999:
                stall += 1
                if stall >= int(cfg.stall_steps):
                    break
            else:
                stall = 0
            prev_err = last_err

        q = clip_qpos_to_limits(
            self.model, self._configuration.q, mujoco_mod=self._mujoco
        )
        self._configuration.update(q=q)
        return q.copy(), last_err, len(active)

    def track_capture(
        self,
        targets: dict[str, np.ndarray | None],
        *,
        qpos_anchor: np.ndarray | None = None,
        **_: object,
    ) -> tuple[np.ndarray, float, int]:
        return self.solve(targets, qpos_init=qpos_anchor)
