"""mink 微分 IK（capture 帧内小步 + limits ladder）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.mujoco_ik.collision_pairs import (
    CollisionAvoidanceConfig,
    DEFAULT_COLLISION_CFG,
    build_collision_avoidance_limit,
)
from core.mujoco_ik.layout import IK_SITE_SPECS, IKSiteSpec, SITE_NAMES
from core.mujoco_ik.qpos_limits import clip_qpos_to_limits
from core.mujoco_ik.root_facing import (
    RootFacingLock,
    apply_root_lock,
    freejoint_v_ids,
    freejoint_rotation_v_ids,
    model_root_on_pelvis,
)


def _joint_names_to_v_ids(model, mujoco_mod, joint_names: tuple[str, ...] | list[str]) -> list[int]:
    """OpenSim/MJCF 关节名 → MuJoCo velocity dof 下标。"""
    try:
        import mink

        dof_width = mink.constants.dof_width
    except ImportError:
        dof_width = lambda t: 1  # noqa: E731 — fallback hinge/slide

    v_ids: list[int] = []
    for name in joint_names:
        jid = mujoco_mod.mj_name2id(model, mujoco_mod.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        vadr = int(model.jnt_dofadr[jid])
        vdim = int(dof_width(int(model.jnt_type[jid])))
        v_ids.extend(range(vadr, vadr + vdim))
    return v_ids


def _keyframe_qpos(model, mujoco_mod) -> np.ndarray | None:
    if model.nkey <= 0:
        return None
    data = mujoco_mod.MjData(model)
    mujoco_mod.mj_resetDataKeyframe(model, data, 0)
    return data.qpos.copy()


@dataclass
class MinkIKConfig:
    """每 capture 帧：`max_steps` 次 solve_ik + integrate(step_dt)。"""

    max_steps: int = 30
    tol: float = 1e-3
    min_targets: int = 4
    step_dt: float = 0.005
    damping: float = 1e-6
    solver: str = "daqp"
    posture_cost: float = 1e-3
    site_weight_scale: float = 1.0
    safety_break: bool = False
    freeze_root_without_pelvis: bool = True
    joint_velocity_limit: float = 4.0
    enable_velocity_limit: bool = True
    stall_steps: int = 5
    enable_collision: bool = True
    collision: CollisionAvoidanceConfig | None = None
    orientation_cost: float = 0.0
    project_targets: bool = True
    allow_empty_limits_fallback: bool = True
    equality_cost: float = 0.0
    root_facing: bool = True
    root_facing_hard_lock: bool = False

    @property
    def max_iter(self) -> int:
        return self.max_steps

    @property
    def dt(self) -> float:
        return self.step_dt


class MinkIKSolver:
    def __init__(
        self,
        model_path: Path | str,
        cfg: MinkIKConfig | None = None,
        *,
        site_specs: tuple[IKSiteSpec, ...] | None = None,
        init_from_keyframe: bool = False,
        root_anchor_sites: tuple[str, ...] | None = None,
        frozen_joint_names: tuple[str, ...] | None = None,
    ) -> None:
        try:
            import mink
        except ImportError as exc:
            raise RuntimeError(
                "未安装 mink。请: pip install mink qpsolvers daqp"
            ) from exc

        import mujoco

        self._mink = mink
        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.cfg = cfg or MinkIKConfig()

        specs = site_specs if site_specs is not None else IK_SITE_SPECS
        self._site_specs = specs
        self._site_names: tuple[str, ...] = tuple(s.site_name for s in specs)
        self._root_anchor_sites = root_anchor_sites or (
            "pelvis",
            "left_hip",
            "right_hip",
        )

        self._site_ids: dict[str, int] = {}
        self._weights: dict[str, float] = {}
        scale = float(self.cfg.site_weight_scale)
        for spec in specs:
            sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, spec.site_name
            )
            if sid < 0:
                raise ValueError(f"MJCF 缺少 site: {spec.site_name}")
            self._site_ids[spec.site_name] = int(sid)
            self._weights[spec.site_name] = float(spec.weight) * scale

        self._joint_names = self._collect_joint_names()
        self._configuration = mink.Configuration(self.model)
        self._root_q_ids, self._root_v_ids = mink.get_freejoint_dims(self.model)
        self._root_v_ids_arr = np.asarray(self._root_v_ids, dtype=int)
        if self._root_v_ids_arr.size == 0:
            self._root_v_ids_arr = freejoint_v_ids(self.model, mujoco)
        self._root_rot_v_ids_arr = freejoint_rotation_v_ids(self.model, mujoco)
        self._frozen_v_ids = _joint_names_to_v_ids(
            self.model, mujoco, frozen_joint_names or ()
        )
        self._init_from_keyframe = bool(init_from_keyframe)
        self._default_qpos = (
            _keyframe_qpos(self.model, mujoco)
            if self._init_from_keyframe
            else None
        )
        if self._default_qpos is None:
            self._default_qpos = self.model.qpos0.copy()

        self._velocity_limits = self._build_velocity_limits()
        self._limit_ladder = self._build_limit_ladder(self.cfg)

        self._posture_task = mink.PostureTask(
            self.model, cost=float(self.cfg.posture_cost)
        )
        self._posture_task.set_target(self._default_qpos.copy())

        self._equality_task: Any | None = None
        eq_cost = float(self.cfg.equality_cost)
        if eq_cost > 0.0 and int(self.model.neq) > 0:
            self._equality_task = mink.EqualityConstraintTask(
                self.model, cost=eq_cost
            )

    def _collect_joint_names(self) -> list[str]:
        names: list[str] = []
        for j in range(self.model.njnt):
            n = self._mujoco.mj_id2name(
                self.model, self._mujoco.mjtObj.mjOBJ_JOINT, j
            )
            names.append(n or f"joint_{j}")
        return names

    def _build_velocity_limits(self) -> dict[str, np.ndarray]:
        if not self.cfg.enable_velocity_limit:
            return {}
        mujoco = self._mujoco
        mink = self._mink
        limits: dict[str, np.ndarray] = {}
        vmax = float(self.cfg.joint_velocity_limit)
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if not name:
                continue
            vdim = mink.constants.dof_width(int(self.model.jnt_type[j]))
            limits[name] = np.full(vdim, vmax, dtype=np.float64)
        return limits

    def _build_limit_ladder(self, cfg: MinkIKConfig) -> list[list[Any]]:
        mink = self._mink
        full: list[Any] = [mink.ConfigurationLimit(self.model)]
        if cfg.enable_velocity_limit and self._velocity_limits:
            full.append(mink.VelocityLimit(self.model, self._velocity_limits))
        if cfg.enable_collision:
            coll_cfg = cfg.collision or DEFAULT_COLLISION_CFG
            full.append(build_collision_avoidance_limit(self.model, coll_cfg))
        ladder: list[list[Any]] = [
            full,
            [mink.ConfigurationLimit(self.model)],
        ]
        if cfg.allow_empty_limits_fallback:
            ladder.append([])
        return ladder

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    def reset(self) -> None:
        self._configuration.update(q=None)

    def _finalize_qpos(self) -> np.ndarray:
        q = clip_qpos_to_limits(
            self.model, self._configuration.q, mujoco_mod=self._mujoco
        )
        self._configuration.update(q=q)
        return q.copy()

    def solve(
        self,
        targets: dict[str, np.ndarray | None],
        *,
        qpos_init: np.ndarray | None = None,
        root_lock: RootFacingLock | None = None,
    ) -> tuple[np.ndarray, float, int]:
        cfg = self.cfg
        active = self._active_targets(targets)
        if len(active) < cfg.min_targets:
            return self._configuration.q.copy(), float("inf"), len(active)

        if qpos_init is not None:
            q_init = np.asarray(qpos_init, dtype=np.float64).copy()
        else:
            q_init = self._default_qpos.copy()
        if root_lock is not None:
            q_init = apply_root_lock(q_init, root_lock)

        self._configuration.update(q=q_init)
        self._posture_task.set_target(q_init)
        freeze_root = (
            (root_lock is not None and root_lock.hard_lock_rotation)
            or self._should_freeze_root(active)
        )

        tasks, constraints = self._build_tasks(
            active,
            freeze_root=freeze_root,
            root_lock=root_lock,
        )
        step_dt = float(cfg.step_dt)
        base_damp = float(cfg.damping)

        last_err = self._weighted_position_residual(active)
        if last_err < cfg.tol:
            q = self._finalize_qpos()
        if root_lock is not None and root_lock.hard_lock_rotation:
            q = apply_root_lock(q, root_lock)
            return q, last_err, len(active)

        stall = 0
        prev_err = last_err
        for _ in range(cfg.max_steps):
            vel = self._solve_velocity(
                tasks,
                constraints,
                self._limit_ladder,
                dt=step_dt,
                base_damping=base_damp,
            )
            if vel is None:
                break

            if root_lock is not None and root_lock.hard_lock_rotation:
                if root_lock.pos is not None:
                    vel[self._root_v_ids_arr] = 0.0
                elif self._root_rot_v_ids_arr.size:
                    vel[self._root_rot_v_ids_arr] = 0.0
            elif freeze_root:
                vel[self._root_v_ids_arr] = 0.0
            if self._frozen_v_ids:
                vel[np.asarray(self._frozen_v_ids, dtype=int)] = 0.0

            self._configuration.integrate_inplace(vel, step_dt)
            if root_lock is not None and root_lock.hard_lock_rotation:
                q_fixed = apply_root_lock(self._configuration.q.copy(), root_lock)
                self._configuration.update(q=q_fixed)

            last_err = self._weighted_position_residual(active)
            if last_err < cfg.tol:
                break
            if last_err >= prev_err * 0.999:
                stall += 1
                if stall >= int(cfg.stall_steps):
                    break
            else:
                stall = 0
            prev_err = last_err

        q = self._finalize_qpos()
        if root_lock is not None and root_lock.hard_lock_rotation:
            q = apply_root_lock(q, root_lock)
            self._configuration.update(q=q)
        return q, last_err, len(active)

    def track_capture(
        self,
        targets: dict[str, np.ndarray | None],
        *,
        targets_prev: dict[str, np.ndarray | None] | None = None,
        qpos_anchor: np.ndarray | None = None,
        reset_session: bool = False,
        root_lock: RootFacingLock | None = None,
    ) -> tuple[np.ndarray, float, int]:
        """session 批处理入口；targets_prev / reset_session 保留兼容。"""
        del targets_prev, reset_session
        return self.solve(targets, qpos_init=qpos_anchor, root_lock=root_lock)

    def _should_freeze_root(self, active: dict[str, np.ndarray]) -> bool:
        cfg = self.cfg
        if not cfg.freeze_root_without_pelvis:
            return False
        has_anchor = any(name in active for name in self._root_anchor_sites)
        return not has_anchor

    def _solve_velocity(
        self,
        tasks: list[Any],
        constraints: list[Any] | None,
        limit_ladder: list[list[Any]],
        *,
        dt: float,
        base_damping: float,
    ) -> np.ndarray | None:
        mink = self._mink
        cfg = self.cfg
        tiers = limit_ladder
        attempts: list[tuple[list[Any], float]] = [
            (tiers[0], base_damping),
            (tiers[0], max(base_damping, 1e-4)),
            (tiers[1], max(base_damping, 1e-4)),
        ]
        if len(tiers) > 2:
            attempts.append((tiers[2], max(base_damping, 1e-2)))

        for limits, damp in attempts:
            try:
                return mink.solve_ik(
                    self._configuration,
                    tasks,
                    dt=dt,
                    solver=cfg.solver,
                    damping=float(damp),
                    limits=limits,
                    constraints=constraints,
                    safety_break=cfg.safety_break,
                )
            except mink.NoSolutionFound:
                continue
        return None

    def _build_tasks(
        self,
        active: dict[str, np.ndarray],
        *,
        freeze_root: bool,
        root_lock: RootFacingLock | None = None,
    ) -> tuple[list[Any], list[Any] | None]:
        mink = self._mink
        cfg = self.cfg
        ori = float(cfg.orientation_cost)

        frame_tasks: list[mink.FrameTask] = []
        for name, pos in active.items():
            w = self._weights[name]
            o_cost = np.full(3, ori * w, dtype=np.float64) if ori > 0.0 else 0.0
            task = mink.FrameTask(
                name,
                "site",
                position_cost=np.full(3, w, dtype=np.float64),
                orientation_cost=o_cost,
            )
            task.set_target(mink.SE3.from_translation(np.asarray(pos, dtype=np.float64)))
            frame_tasks.append(task)

        tasks: list[Any] = frame_tasks + [self._posture_task]
        if self._equality_task is not None:
            tasks.append(self._equality_task)

        constraints: list[Any] | None = None
        freeze_v_ids: list[int] = []
        if root_lock is not None and root_lock.hard_lock_rotation:
            if root_lock.pos is not None and self._root_v_ids:
                freeze_v_ids.extend(int(v) for v in self._root_v_ids)
            elif self._root_rot_v_ids_arr.size:
                freeze_v_ids.extend(int(v) for v in self._root_rot_v_ids_arr)
        elif freeze_root and self._root_v_ids:
            freeze_v_ids.extend(int(v) for v in self._root_v_ids)
        if self._frozen_v_ids:
            freeze_v_ids.extend(int(v) for v in self._frozen_v_ids)
        if freeze_v_ids:
            constraints = [
                mink.DofFreezingTask(
                    self.model,
                    list(dict.fromkeys(freeze_v_ids)),
                )
            ]

        return tasks, constraints

    def site_positions(self) -> dict[str, np.ndarray]:
        self._mujoco.mj_forward(self.model, self._configuration.data)
        return {
            name: self._configuration.data.site_xpos[sid].copy()
            for name, sid in self._site_ids.items()
        }

    def _weighted_position_residual(self, active: dict[str, np.ndarray]) -> float:
        self._mujoco.mj_forward(self.model, self._configuration.data)
        errs: list[np.ndarray] = []
        for name, tgt in active.items():
            sid = self._site_ids[name]
            cur = self._configuration.data.site_xpos[sid]
            w = self._weights[name]
            errs.append((tgt - cur) * w)
        if not errs:
            return float("inf")
        return float(np.linalg.norm(np.concatenate(errs)))

    def _active_targets(
        self, targets: dict[str, np.ndarray | None]
    ) -> dict[str, np.ndarray]:
        active: dict[str, np.ndarray] = {}
        for name in self._site_names:
            p = targets.get(name)
            if p is None:
                continue
            arr = np.asarray(p, dtype=np.float64).reshape(3)
            if np.all(np.isfinite(arr)) and np.linalg.norm(arr) > 1e-4:
                active[name] = arr
        return active
