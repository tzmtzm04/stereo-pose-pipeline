"""MuJoCo qpos 硬限位：hinge/slide 读 MJCF ``jnt_range``（有正负）。"""

from __future__ import annotations

import numpy as np


def _hinge_slide_joints(model, mujoco_mod) -> list[tuple[int, float, float]]:
    """(qposadr, lo, hi) for limited hinge/slide joints."""
    hinge = int(mujoco_mod.mjtJoint.mjJNT_HINGE)
    slide = int(mujoco_mod.mjtJoint.mjJNT_SLIDE)
    out: list[tuple[int, float, float]] = []
    for j in range(int(model.njnt)):
        if not model.jnt_limited[j]:
            continue
        jtype = int(model.jnt_type[j])
        if jtype not in (hinge, slide):
            continue
        adr = int(model.jnt_qposadr[j])
        lo, hi = float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1])
        out.append((adr, lo, hi))
    return out


def clip_qpos_to_limits(model, qpos: np.ndarray, *, mujoco_mod) -> np.ndarray:
    """将 qpos 中 limited hinge/slide 分量 clip 到 ``model.jnt_range``。"""
    q = np.asarray(qpos, dtype=np.float64).copy()
    for adr, lo, hi in _hinge_slide_joints(model, mujoco_mod):
        q[adr] = float(np.clip(q[adr], lo, hi))
    return q


def hinges_at_bound(
    model,
    qpos: np.ndarray,
    *,
    mujoco_mod,
    eps: float = 1e-3,
) -> tuple[list[int], list[int]]:
    """返回贴 lower / upper 的 hinge qpos 下标。"""
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    at_lo: list[int] = []
    at_hi: list[int] = []
    for adr, lo, hi in _hinge_slide_joints(model, mujoco_mod):
        v = float(q[adr])
        if v <= lo + eps:
            at_lo.append(adr)
        elif v >= hi - eps:
            at_hi.append(adr)
    return at_lo, at_hi


def nudge_hinges_off_bounds(
    model,
    qpos: np.ndarray,
    *,
    mujoco_mod,
    lower_frac: float = 0.25,
    upper_frac: float = 0.25,
    eps: float = 1e-3,
) -> np.ndarray:
    """将贴边的 hinge 推入 range 内部（用于多初值 escape，不绑定具体关节名）。"""
    q = np.asarray(qpos, dtype=np.float64).copy()
    lo_frac = float(np.clip(lower_frac, 0.05, 0.95))
    hi_frac = float(np.clip(upper_frac, 0.05, 0.95))
    for adr, lo, hi in _hinge_slide_joints(model, mujoco_mod):
        span = hi - lo
        if span < 1e-9:
            continue
        v = float(q[adr])
        if v <= lo + eps:
            q[adr] = lo + lo_frac * span
        elif v >= hi - eps:
            q[adr] = hi - hi_frac * span
    return clip_qpos_to_limits(model, q, mujoco_mod=mujoco_mod)
