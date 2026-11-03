"""阻尼最小二乘 Jacobian IK（MuJoCo mj_jacSite）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.mujoco_ik.layout import IK_SITE_SPECS, IKSiteSpec, SITE_NAMES
from core.mujoco_ik.qpos_limits import (
    clip_qpos_to_limits,
    hinges_at_bound,
    nudge_hinges_off_bounds,
)
from core.mujoco_ik.root_facing import (
    RootFacingLock,
    apply_root_lock,
    freejoint_rotation_v_ids,
    freejoint_v_ids,
)


@dataclass
class JacobianIKConfig:
    max_iter: int = 50
    damping: float = 0.05
    step_scale: float = 0.85
    tol: float = 1e-3
    min_targets: int = 4
    dq_clip: float = 0.25
    multi_init: bool = True
    hinge_bound_eps: float = 1e-3
    hinge_escape_frac: float = 0.25
    root_facing: bool = True
    root_facing_hard_lock: bool = False


class JacobianIKSolver:
    def __init__(
        self,
        model_path: Path | str,
        cfg: JacobianIKConfig | None = None,
        *,
        site_specs: tuple[IKSiteSpec, ...] | None = None,
        init_from_keyframe: bool = False,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self.cfg = cfg or JacobianIKConfig()
        self._init_from_keyframe = bool(init_from_keyframe)

        specs = site_specs if site_specs is not None else IK_SITE_SPECS
        self._site_specs = specs
        self._site_names: tuple[str, ...] = tuple(s.site_name for s in specs)

        self._site_ids: dict[str, int] = {}
        self._weights: dict[str, float] = {}
        for spec in specs:
            sid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, spec.site_name
            )
            if sid < 0:
                raise ValueError(f"MJCF 缺少 site: {spec.site_name}")
            self._site_ids[spec.site_name] = int(sid)
            self._weights[spec.site_name] = float(spec.weight)

        self._joint_names = self._collect_joint_names()
        self._root_v_ids = freejoint_v_ids(self.model, mujoco)
        self._root_rot_v_ids = freejoint_rotation_v_ids(self.model, mujoco)

    def _collect_joint_names(self) -> list[str]:
        names: list[str] = []
        for j in range(self.model.njnt):
            n = self._mujoco.mj_id2name(
                self.model, self._mujoco.mjtObj.mjOBJ_JOINT, j
            )
            names.append(n or f"joint_{j}")
        return names

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    def reset(self) -> None:
        if self._init_from_keyframe and self.model.nkey > 0:
            self._mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        else:
            self._mujoco.mj_resetData(self.model, self.data)

    def _default_qpos(self) -> np.ndarray:
        self.reset()
        return clip_qpos_to_limits(
            self.model, self.data.qpos.copy(), mujoco_mod=self._mujoco
        )

    def _init_candidates(
        self,
        qpos_init: np.ndarray | None,
        *,
        root_lock: RootFacingLock | None = None,
    ) -> list[np.ndarray]:
        cfg = self.cfg
        mujoco = self._mujoco
        if root_lock is not None and root_lock.hard_lock_rotation:
            base = (
                apply_root_lock(qpos_init, root_lock)
                if qpos_init is not None
                else apply_root_lock(self._default_qpos(), root_lock)
            )
            return [clip_qpos_to_limits(self.model, base, mujoco_mod=mujoco)]

        if qpos_init is not None:
            primary = clip_qpos_to_limits(
                self.model, qpos_init, mujoco_mod=mujoco
            )
        else:
            primary = self._default_qpos()

        if not cfg.multi_init:
            return [primary]

        cands: list[np.ndarray] = [primary]
        escaped = nudge_hinges_off_bounds(
            self.model,
            primary,
            mujoco_mod=mujoco,
            lower_frac=cfg.hinge_escape_frac,
            upper_frac=cfg.hinge_escape_frac,
            eps=cfg.hinge_bound_eps,
        )
        if not np.allclose(escaped, primary, atol=1e-8):
            cands.append(escaped)

        neutral = self._default_qpos()
        if not any(np.allclose(neutral, c, atol=1e-8) for c in cands):
            cands.append(neutral)
        return cands

    def solve(
        self,
        targets: dict[str, np.ndarray | None],
        *,
        qpos_init: np.ndarray | None = None,
        root_lock: RootFacingLock | None = None,
    ) -> tuple[np.ndarray, float, int]:
        """返回 (qpos, 残差范数, 有效目标数)。"""
        active = self._active_targets(targets)
        if len(active) < self.cfg.min_targets:
            q = (
                clip_qpos_to_limits(
                    self.model,
                    np.asarray(qpos_init, dtype=np.float64),
                    mujoco_mod=self._mujoco,
                )
                if qpos_init is not None
                else self._default_qpos()
            )
            return q, float("inf"), len(active)

        best_q: np.ndarray | None = None
        best_err = float("inf")
        n_tgt = len(active)

        q_seed = qpos_init
        if root_lock is not None and q_seed is not None:
            q_seed = apply_root_lock(np.asarray(q_seed, dtype=np.float64), root_lock)

        for q0 in self._init_candidates(q_seed, root_lock=root_lock):
            q, err = self._solve_once(active, q0, root_lock=root_lock)
            if err < best_err:
                best_err = err
                best_q = q
            if err < self.cfg.tol:
                break
            at_lo, _at_hi = hinges_at_bound(
                self.model,
                q,
                mujoco_mod=self._mujoco,
                eps=self.cfg.hinge_bound_eps,
            )
            if not at_lo and err < self.cfg.tol * 10.0:
                break

        assert best_q is not None
        return best_q, best_err, n_tgt

    def _solve_once(
        self,
        active: dict[str, np.ndarray],
        qpos_init: np.ndarray,
        *,
        root_lock: RootFacingLock | None = None,
    ) -> tuple[np.ndarray, float]:
        mujoco = self._mujoco
        cfg = self.cfg
        q0 = np.asarray(qpos_init, dtype=np.float64)
        if root_lock is not None:
            q0 = apply_root_lock(q0, root_lock)
        self.data.qpos[:] = q0

        last_err = float("inf")
        for _ in range(cfg.max_iter):
            mujoco.mj_forward(self.model, self.data)
            rows_j: list[np.ndarray] = []
            errs: list[np.ndarray] = []
            for name, tgt in active.items():
                sid = self._site_ids[name]
                w = self._weights[name]
                cur = self.data.site_xpos[sid].copy()
                err = (tgt - cur) * w
                jacp = np.zeros((3, self.model.nv), dtype=np.float64)
                jacr = np.zeros((3, self.model.nv), dtype=np.float64)
                mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
                rows_j.append(jacp * w)
                errs.append(err)

            e = np.concatenate(errs)
            last_err = float(np.linalg.norm(e))
            if last_err < cfg.tol:
                break

            j_mat = np.vstack(rows_j)
            n = j_mat.shape[0]
            a_mat = j_mat @ j_mat.T + (cfg.damping**2) * np.eye(n, dtype=np.float64)
            dq = j_mat.T @ np.linalg.solve(a_mat, e)
            if (
                root_lock is not None
                and root_lock.hard_lock_rotation
                and self._root_rot_v_ids.size
            ):
                dq[self._root_rot_v_ids] = 0.0
            elif (
                root_lock is not None
                and root_lock.hard_lock_rotation
                and self._root_v_ids.size
            ):
                dq[self._root_v_ids] = 0.0
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > cfg.dq_clip:
                dq *= cfg.dq_clip / dq_norm
            mujoco.mj_integratePos(
                self.model, self.data.qpos, dq, cfg.step_scale
            )
            self.data.qpos[:] = clip_qpos_to_limits(
                self.model, self.data.qpos, mujoco_mod=mujoco
            )
            if root_lock is not None and root_lock.hard_lock_rotation:
                self.data.qpos[:] = apply_root_lock(self.data.qpos, root_lock)

        q = clip_qpos_to_limits(
            self.model, self.data.qpos.copy(), mujoco_mod=mujoco
        )
        if root_lock is not None and root_lock.hard_lock_rotation:
            q = apply_root_lock(q, root_lock)
        return q, last_err

    def site_positions(self) -> dict[str, np.ndarray]:
        self._mujoco.mj_forward(self.model, self.data)
        return {
            name: self.data.site_xpos[sid].copy()
            for name, sid in self._site_ids.items()
        }

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
