"""IK 前 / Stage 4 3D 目标投影：骨长 FABRIK + 躯干避让。

肘/膝角限默认在 IK 层（MJCF hinge ``range`` + mink ConfigurationLimit / qpos clip），
不在此做无符号几何角限。``angle_first`` / ``enable_joint_limits`` 保留作 legacy。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from core.mujoco_ik.keypoint_constraints import (
    JointLimitRule,
    apply_joint_limits,
    solve_two_bone_mid,
    wb133_joint_limit_rules,
)
from core.mujoco_ik.layout import SITE_NAMES

ProjectionPriority = Literal["bone_first", "angle_first"]


@dataclass(frozen=True)
class TargetProjectionConfig:
    """与 wb133_ik_humanoid.xml 骨段长度一致。"""

    upper_arm_m: float = 0.26
    forearm_m: float = 0.24
    thigh_m: float = 0.40
    shin_m: float = 0.40
    torso_capsule_radius_m: float = 0.10
    torso_clearance_m: float = 0.05
    max_point_shift_m: float = 0.25
    blend_prev: float = 0.0
    hard_projection: bool = False
    enable_joint_limits: bool = False
    elbow_min_deg: float = 25.0
    elbow_max_deg: float = 175.0
    knee_min_deg: float = 20.0
    knee_max_deg: float = 175.0
    refit_after_joint_limits: bool = True
    projection_priority: ProjectionPriority = "bone_first"
    refit_arms_after_torso: bool = True


# (root, tip, max_len_m) — 仅 soft 模式下的两段裁剪
_LIMB_SEGMENTS: tuple[tuple[str, str, str], ...] = (
    ("left_shoulder", "left_elbow", "upper_arm_m"),
    ("right_shoulder", "right_elbow", "upper_arm_m"),
    ("left_hip", "left_knee", "thigh_m"),
    ("right_hip", "right_knee", "thigh_m"),
)

# (root, mid, tip, len_ab, len_bc)
_LIMB_CHAINS: tuple[tuple[str, str, str, str, str], ...] = (
    ("left_shoulder", "left_elbow", "left_wrist", "upper_arm_m", "forearm_m"),
    ("right_shoulder", "right_elbow", "right_wrist", "upper_arm_m", "forearm_m"),
    ("left_hip", "left_knee", "left_ankle", "thigh_m", "shin_m"),
    ("right_hip", "right_knee", "right_ankle", "thigh_m", "shin_m"),
)

# (root, mid, tip, len_ab, len_bc, min_deg_attr, max_deg_attr)
_LIMB_CHAINS_ANGLE: tuple[tuple[str, str, str, str, str, str, str], ...] = (
    (
        "left_shoulder",
        "left_elbow",
        "left_wrist",
        "upper_arm_m",
        "forearm_m",
        "elbow_min_deg",
        "elbow_max_deg",
    ),
    (
        "right_shoulder",
        "right_elbow",
        "right_wrist",
        "upper_arm_m",
        "forearm_m",
        "elbow_min_deg",
        "elbow_max_deg",
    ),
    (
        "left_hip",
        "left_knee",
        "left_ankle",
        "thigh_m",
        "shin_m",
        "knee_min_deg",
        "knee_max_deg",
    ),
    (
        "right_hip",
        "right_knee",
        "right_ankle",
        "thigh_m",
        "shin_m",
        "knee_min_deg",
        "knee_max_deg",
    ),
)

_ARM_CHAIN_KEYS: frozenset[tuple[str, str, str]] = frozenset(
    (root_n, mid_n, tip_n)
    for root_n, mid_n, tip_n, *_ in _LIMB_CHAINS_ANGLE
    if "shoulder" in root_n
)

_TORSO_PUSH_SITES = (
    "left_elbow",
    "right_elbow",
)


def _get(
    targets: dict[str, np.ndarray | None], name: str
) -> np.ndarray | None:
    p = targets.get(name)
    if p is None:
        return None
    arr = np.asarray(p, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def _set(targets: dict[str, np.ndarray | None], name: str, p: np.ndarray) -> None:
    targets[name] = np.asarray(p, dtype=np.float64).reshape(3).copy()


def _clip_segment(
    root: np.ndarray, tip: np.ndarray, max_len: float
) -> np.ndarray:
    v = tip - root
    d = float(np.linalg.norm(v))
    if d <= max_len or d < 1e-9:
        return tip.copy()
    return root + v * (max_len / d)


def _fabrik3(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    lab: float,
    lbc: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """单遍 FABRIK：固定 root，调整 mid/tip 满足骨长。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)

    bc = c - b
    bc_len = float(np.linalg.norm(bc))
    if bc_len > lbc:
        c = b + bc * (lbc / bc_len)

    ac = c - a
    ac_len = float(np.linalg.norm(ac))
    reach = lab + lbc
    if ac_len > reach and ac_len > 1e-9:
        c = a + ac * (reach / ac_len)
        b = a + (c - a) * (lab / reach)
    else:
        ab = b - a
        ab_len = float(np.linalg.norm(ab))
        if ab_len > lab and ab_len > 1e-9:
            b = a + ab * (lab / ab_len)
        bc = c - b
        bc_len = float(np.linalg.norm(bc))
        if bc_len > lbc and bc_len > 1e-9:
            c = b + bc * (lbc / bc_len)

    return a, b, c


def _joint_angle_rad(
    root: np.ndarray,
    joint: np.ndarray,
    tip: np.ndarray,
) -> float:
    """∠(root-joint-tip) [rad]。"""
    ba = np.asarray(root, dtype=np.float64) - np.asarray(joint, dtype=np.float64)
    bc = np.asarray(tip, dtype=np.float64) - np.asarray(joint, dtype=np.float64)
    la = float(np.linalg.norm(ba))
    lc = float(np.linalg.norm(bc))
    if la < 1e-9 or lc < 1e-9:
        return math.pi * 0.5
    cos_a = float(np.clip(np.dot(ba / la, bc / lc), -1.0, 1.0))
    return math.acos(cos_a)


def _project_limb_angle_first(
    root: np.ndarray,
    mid: np.ndarray,
    tip: np.ndarray,
    lab: float,
    lbc: float,
    min_deg: float,
    max_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """角限优先：clip θ → 由余弦定理定 |root-tip| → 两节 IK 求 mid。"""
    a = np.asarray(root, dtype=np.float64)
    b = np.asarray(mid, dtype=np.float64)
    c = np.asarray(tip, dtype=np.float64)

    theta = _joint_angle_rad(a, b, c)
    min_r = math.radians(float(min_deg))
    max_r = math.radians(float(max_deg))
    theta = float(np.clip(theta, min_r, max_r))

    cos_t = math.cos(theta)
    d_sw_sq = lab * lab + lbc * lbc - 2.0 * lab * lbc * cos_t
    d_sw = math.sqrt(max(0.0, d_sw_sq))

    sw = c - a
    d_obs = float(np.linalg.norm(sw))
    if d_obs < 1e-9:
        sw = b - a
        d_obs = float(np.linalg.norm(sw))
    if d_obs < 1e-9:
        sw_u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        sw_u = sw / d_obs

    c2 = a + sw_u * d_sw
    b2 = solve_two_bone_mid(a, c2, float(lab), float(lbc), b - a)
    return a, b2, c2


def max_bone_violation_m(
    targets: dict[str, np.ndarray | None],
    cfg: TargetProjectionConfig,
) -> float:
    worst = 0.0
    for root_n, mid_n, tip_n, len_ab, len_bc in _LIMB_CHAINS:
        a = _get(targets, root_n)
        b = _get(targets, mid_n)
        c = _get(targets, tip_n)
        if a is None or b is None or c is None:
            continue
        lab = float(getattr(cfg, len_ab))
        lbc = float(getattr(cfg, len_bc))
        ab = float(np.linalg.norm(b - a))
        bc = float(np.linalg.norm(c - b))
        worst = max(worst, ab - lab, bc - lbc)
    return max(0.0, worst)


def max_angle_violation_deg(
    targets: dict[str, np.ndarray | None],
    cfg: TargetProjectionConfig,
) -> float:
    worst = 0.0
    for root_n, mid_n, tip_n, _, _, min_attr, max_attr in _LIMB_CHAINS_ANGLE:
        a = _get(targets, root_n)
        b = _get(targets, mid_n)
        c = _get(targets, tip_n)
        if a is None or b is None or c is None:
            continue
        theta = math.degrees(_joint_angle_rad(a, b, c))
        lo = float(getattr(cfg, min_attr))
        hi = float(getattr(cfg, max_attr))
        worst = max(worst, lo - theta, theta - hi)
    return max(0.0, worst)


def _closest_on_segment(
    p: np.ndarray, a: np.ndarray, b: np.ndarray
) -> tuple[np.ndarray, float]:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-12:
        q = a.copy()
        return q, float(np.linalg.norm(p - q))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    q = a + t * ab
    return q, float(np.linalg.norm(p - q))


def _push_outside_capsule(
    p: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    radius: float,
    clearance: float,
) -> np.ndarray:
    q, dist_axis = _closest_on_segment(p, a, b)
    need = radius + clearance
    if dist_axis >= need:
        return p.copy()
    outward = p - q
    out_len = float(np.linalg.norm(outward))
    if out_len < 1e-6:
        outward = p - a
        out_len = float(np.linalg.norm(outward))
    if out_len < 1e-6:
        outward = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        out_len = 1.0
    return q + outward * (need / out_len)


def _limit_shift(orig: np.ndarray, moved: np.ndarray, max_shift: float) -> np.ndarray:
    d = moved - orig
    n = float(np.linalg.norm(d))
    if n <= max_shift or n < 1e-9:
        return moved
    return orig + d * (max_shift / n)


class TargetConstraintProjector:
    """将观测 3D 点投影到骨长 + 躯干胶囊 + 肘/膝几何角限可行域。"""

    def __init__(
        self,
        model_path: Path | str | None = None,
        cfg: TargetProjectionConfig | None = None,
    ) -> None:
        del model_path  # 骨长/躯干用解析几何；保留参数便于日后 FK 投影
        self.cfg = cfg or TargetProjectionConfig()

    def project(
        self,
        targets: dict[str, np.ndarray | None],
        *,
        qpos_hint: np.ndarray | None = None,
        targets_prev: dict[str, np.ndarray | None] | None = None,
    ) -> dict[str, np.ndarray | None]:
        del qpos_hint
        out: dict[str, np.ndarray | None] = {
            name: (None if targets.get(name) is None else np.asarray(targets[name], dtype=np.float64).reshape(3).copy())
            for name in SITE_NAMES
        }

        if self.cfg.projection_priority == "angle_first":
            self._project_limb_chains_angle_first(out)
            self._project_torso_clearance(out)
            if self.cfg.refit_arms_after_torso:
                self._project_limb_chains_angle_first(out, arms_only=True)
        else:
            self._project_limb_chains(out)
            if not self.cfg.hard_projection:
                self._project_bone_segments(out)
            self._project_torso_clearance(out)
            self._project_joint_limits(out)
            if self.cfg.enable_joint_limits and self.cfg.refit_after_joint_limits:
                self._project_limb_chains(out)

        if targets_prev is not None and self.cfg.blend_prev > 0.0:
            alpha = float(np.clip(self.cfg.blend_prev, 0.0, 1.0))
            for name in SITE_NAMES:
                cur = out.get(name)
                prev = targets_prev.get(name) if targets_prev else None
                if cur is None or prev is None:
                    continue
                prev_a = np.asarray(prev, dtype=np.float64).reshape(3)
                blended = (1.0 - alpha) * cur + alpha * prev_a
                out[name] = _limit_shift(cur, blended, self.cfg.max_point_shift_m)

        return out

    def _apply_point(
        self, targets: dict[str, np.ndarray | None], name: str, orig: np.ndarray, moved: np.ndarray
    ) -> None:
        if self.cfg.hard_projection:
            _set(targets, name, moved)
        else:
            _set(targets, name, _limit_shift(orig, moved, self.cfg.max_point_shift_m))

    def _project_limb_chains(self, targets: dict[str, np.ndarray | None]) -> None:
        cfg = self.cfg
        for root_n, mid_n, tip_n, len_ab, len_bc in _LIMB_CHAINS:
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

    def _project_limb_chains_angle_first(
        self,
        targets: dict[str, np.ndarray | None],
        *,
        arms_only: bool = False,
    ) -> None:
        cfg = self.cfg
        for root_n, mid_n, tip_n, len_ab, len_bc, min_attr, max_attr in _LIMB_CHAINS_ANGLE:
            if arms_only and (root_n, mid_n, tip_n) not in _ARM_CHAIN_KEYS:
                continue
            a = _get(targets, root_n)
            b = _get(targets, mid_n)
            c = _get(targets, tip_n)
            if a is None or b is None or c is None:
                continue
            lab = float(getattr(cfg, len_ab))
            lbc = float(getattr(cfg, len_bc))
            lo = float(getattr(cfg, min_attr))
            hi = float(getattr(cfg, max_attr))
            _a, b2, c2 = _project_limb_angle_first(a, b, c, lab, lbc, lo, hi)
            b_orig, c_orig = b.copy(), c.copy()
            self._apply_point(targets, mid_n, b_orig, b2)
            self._apply_point(targets, tip_n, c_orig, c2)

    def _project_bone_segments(self, targets: dict[str, np.ndarray | None]) -> None:
        cfg = self.cfg
        for root_n, tip_n, len_attr in _LIMB_SEGMENTS:
            root = _get(targets, root_n)
            tip = _get(targets, tip_n)
            if root is None or tip is None:
                continue
            max_len = float(getattr(cfg, len_attr))
            orig_tip = tip.copy()
            clipped = _clip_segment(root, tip, max_len)
            self._apply_point(targets, tip_n, orig_tip, clipped)

    def _project_torso_clearance(self, targets: dict[str, np.ndarray | None]) -> None:
        pelvis = _get(targets, "pelvis")
        if pelvis is None:
            return
        ls = _get(targets, "left_shoulder")
        rs = _get(targets, "right_shoulder")
        if ls is not None and rs is not None:
            top = 0.5 * (ls + rs)
        else:
            top = pelvis + np.array([0.0, 0.0, 0.34], dtype=np.float64)

        cfg = self.cfg
        for name in _TORSO_PUSH_SITES:
            p = _get(targets, name)
            if p is None:
                continue
            moved = _push_outside_capsule(
                p,
                pelvis,
                top,
                cfg.torso_capsule_radius_m,
                cfg.torso_clearance_m,
            )
            self._apply_point(targets, name, p, moved)

    def _joint_limit_rules(self) -> tuple[JointLimitRule, ...]:
        cfg = self.cfg
        return wb133_joint_limit_rules(
            elbow_min_deg=float(cfg.elbow_min_deg),
            elbow_max_deg=float(cfg.elbow_max_deg),
            knee_min_deg=float(cfg.knee_min_deg),
            knee_max_deg=float(cfg.knee_max_deg),
        )

    def _project_joint_limits(
        self, targets: dict[str, np.ndarray | None]
    ) -> None:
        if not self.cfg.enable_joint_limits:
            return
        rules = self._joint_limit_rules()
        if self.cfg.hard_projection:
            apply_joint_limits(targets, rules)
            return
        orig: dict[str, np.ndarray] = {}
        for rule in rules:
            if not rule.move_tip:
                continue
            p = _get(targets, rule.tip)
            if p is not None:
                orig[rule.tip] = p.copy()
        apply_joint_limits(targets, rules)
        for name, o in orig.items():
            cur = _get(targets, name)
            if cur is not None:
                self._apply_point(targets, name, o, cur)
