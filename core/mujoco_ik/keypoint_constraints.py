"""关键点层约束（自 mink Limit 择出，不依赖 qpos / IK）。

近似映射：
  CollisionAvoidanceLimit → 点/骨段胶囊最小间距
  ConfigurationLimit      → 肘/膝铰链角（在 IK 层 enforce；此处几何角限为 legacy 可选）
  VelocityLimit           → 可选逐帧最大位移（在 session 层用 targets_prev）
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from core.mujoco_ik.layout import SITE_NAMES

SiteDict = dict[str, np.ndarray | None]


@dataclass(frozen=True)
class SiteSegment:
    """命名 site 间的骨段胶囊（半径 + 可移动端）。"""

    a: str
    b: str
    radius_m: float
    movable: str  # 碰撞时优先推此端（'a' | 'b' | 'both'）


@dataclass(frozen=True)
class PointPointRule:
    """两点最小间距（对应 mink 短 geom 间 clearance）。"""

    p: str
    q: str
    min_dist_m: float
    movable_p: bool = True
    movable_q: bool = True


@dataclass(frozen=True)
class PointSegmentRule:
    """点 ↔ 骨段胶囊最小间距。"""

    point: str
    seg: SiteSegment
    clearance_m: float
    movable_point: bool = True


@dataclass(frozen=True)
class JointLimitRule:
    """三节铰链角（度）：∠(root-joint-tip)。"""

    root: str
    joint: str
    tip: str
    min_deg: float
    max_deg: float
    move_tip: bool = True


# 与 collision_pairs.py geom 组对应的 site 骨段
LEFT_UPPER_ARM = SiteSegment("left_shoulder", "left_elbow", 0.05, "b")
LEFT_FOREARM = SiteSegment("left_elbow", "left_wrist", 0.05, "b")
RIGHT_UPPER_ARM = SiteSegment("right_shoulder", "right_elbow", 0.05, "b")
RIGHT_FOREARM = SiteSegment("right_elbow", "right_wrist", 0.05, "b")
LEFT_THIGH = SiteSegment("left_hip", "left_knee", 0.06, "b")
LEFT_SHIN = SiteSegment("left_knee", "left_ankle", 0.055, "b")
RIGHT_THIGH = SiteSegment("right_hip", "right_knee", 0.06, "b")
RIGHT_SHIN = SiteSegment("right_knee", "right_ankle", 0.055, "b")

# mink WB133_IK_COLLISION_PAIRS → 关键点规则（见 collision_pairs.py 注释）
POINT_SEGMENT_CLEARANCE_RULES: tuple[PointSegmentRule, ...] = (
    # 前臂 ↔ 对侧腿
    PointSegmentRule("left_wrist", RIGHT_THIGH, 0.03),
    PointSegmentRule("left_wrist", RIGHT_SHIN, 0.03),
    PointSegmentRule("right_wrist", LEFT_THIGH, 0.03),
    PointSegmentRule("right_wrist", LEFT_SHIN, 0.03),
    # 同侧前臂 ↔ 同侧腿
    PointSegmentRule("left_wrist", LEFT_THIGH, 0.03),
    PointSegmentRule("left_wrist", LEFT_SHIN, 0.03),
    PointSegmentRule("right_wrist", RIGHT_THIGH, 0.03),
    PointSegmentRule("right_wrist", RIGHT_SHIN, 0.03),
)

POINT_POINT_CLEARANCE_RULES: tuple[PointPointRule, ...] = (
    # 左右臂 / 腿 交叉（geom 组间 minimum_distance ≈ 0.03）
    PointPointRule("left_wrist", "right_wrist", 0.06, True, True),
    PointPointRule("left_elbow", "right_elbow", 0.06, True, True),
    PointPointRule("left_knee", "right_knee", 0.06, True, True),
    PointPointRule("left_ankle", "right_ankle", 0.05, True, True),
)

# Legacy：无符号几何 ∠(root-joint-tip)，近似 ConfigurationLimit（默认不在 Stage 4 使用）
JOINT_LIMIT_RULES: tuple[JointLimitRule, ...] = (
    JointLimitRule("left_shoulder", "left_elbow", "left_wrist", 25.0, 175.0),
    JointLimitRule("right_shoulder", "right_elbow", "right_wrist", 25.0, 175.0),
    JointLimitRule("left_hip", "left_knee", "left_ankle", 20.0, 175.0),
    JointLimitRule("right_hip", "right_knee", "right_ankle", 20.0, 175.0),
)


def wb133_joint_limit_rules(
    *,
    elbow_min_deg: float = 25.0,
    elbow_max_deg: float = 175.0,
    knee_min_deg: float = 20.0,
    knee_max_deg: float = 175.0,
) -> tuple[JointLimitRule, ...]:
    """wb133 / Stage 4：肩–肘–腕、髋–膝–踝几何角限（仅动 tip）。"""
    return (
        JointLimitRule(
            "left_shoulder", "left_elbow", "left_wrist",
            elbow_min_deg, elbow_max_deg,
        ),
        JointLimitRule(
            "right_shoulder", "right_elbow", "right_wrist",
            elbow_min_deg, elbow_max_deg,
        ),
        JointLimitRule(
            "left_hip", "left_knee", "left_ankle",
            knee_min_deg, knee_max_deg,
        ),
        JointLimitRule(
            "right_hip", "right_knee", "right_ankle",
            knee_min_deg, knee_max_deg,
        ),
    )

_WRIST_TORSO_SITES = ("left_wrist", "right_wrist")
_ELBOW_TORSO_SITES = ("left_elbow", "right_elbow")
_ARM_CHAINS: tuple[tuple[str, str, str], ...] = (
    ("left_shoulder", "left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow", "right_wrist"),
)


def get_site(targets: SiteDict, name: str) -> np.ndarray | None:
    p = targets.get(name)
    if p is None:
        return None
    arr = np.asarray(p, dtype=np.float64).reshape(3)
    if not np.all(np.isfinite(arr)):
        return None
    return arr


def set_site(targets: SiteDict, name: str, p: np.ndarray) -> None:
    targets[name] = np.asarray(p, dtype=np.float64).reshape(3).copy()


def _segment_endpoints(
    targets: SiteDict, seg: SiteSegment
) -> tuple[np.ndarray, np.ndarray] | None:
    a = get_site(targets, seg.a)
    b = get_site(targets, seg.b)
    if a is None or b is None:
        return None
    return a, b


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


def push_point_outside_capsule(
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


def push_points_apart(
    p: np.ndarray,
    q: np.ndarray,
    min_dist: float,
    *,
    move_p: bool = True,
    move_q: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    delta = q - p
    dist = float(np.linalg.norm(delta))
    if dist >= min_dist or dist < 1e-9:
        return p.copy(), q.copy()
    need = 0.5 * (min_dist - dist)
    direction = delta / dist
    p2, q2 = p.copy(), q.copy()
    if move_p and move_q:
        p2 = p - direction * need
        q2 = q + direction * need
    elif move_p:
        p2 = p - direction * (min_dist - dist)
    elif move_q:
        q2 = q + direction * (min_dist - dist)
    return p2, q2


def _perpendicular_unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    v = v / n
    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(v, helper))) > 0.95:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = np.cross(v, helper)
    un = float(np.linalg.norm(u))
    if un < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    return u / un


def clamp_joint_angle(
    root: np.ndarray,
    joint: np.ndarray,
    tip: np.ndarray,
    *,
    min_deg: float,
    max_deg: float,
) -> np.ndarray:
    """硬限位：调整 tip，使 ∠(root-joint-tip) 落在 [min_deg, max_deg]。"""
    ba = np.asarray(root, dtype=np.float64) - joint
    bc = np.asarray(tip, dtype=np.float64) - joint
    la = float(np.linalg.norm(ba))
    lc = float(np.linalg.norm(bc))
    if la < 1e-9 or lc < 1e-9:
        return tip.copy()

    ba_u = ba / la
    bc_u = bc / lc
    cos_a = float(np.clip(np.dot(ba_u, bc_u), -1.0, 1.0))
    angle = math.acos(cos_a)
    min_r = math.radians(float(min_deg))
    max_r = math.radians(float(max_deg))
    if min_r <= angle <= max_r:
        return tip.copy()

    target = min_r if angle < min_r else max_r
    axis = np.cross(ba_u, bc_u)
    axn = float(np.linalg.norm(axis))
    if axn < 1e-6:
        axis = _perpendicular_unit(ba_u)
    else:
        axis = axis / axn

    delta = target - angle
    c, s = math.cos(delta), math.sin(delta)
    bc_rot = (
        bc_u * c
        + np.cross(axis, bc_u) * s
        + axis * float(np.dot(axis, bc_u)) * (1.0 - c)
    )
    bn = float(np.linalg.norm(bc_rot))
    if bn < 1e-9:
        return tip.copy()
    return joint + (bc_rot / bn) * lc


def head_proxy(targets: SiteDict, offset_m: float) -> np.ndarray | None:
    pelvis = get_site(targets, "pelvis")
    ls = get_site(targets, "left_shoulder")
    rs = get_site(targets, "right_shoulder")
    if pelvis is None:
        return None
    if ls is not None and rs is not None:
        mid = 0.5 * (ls + rs)
    else:
        mid = pelvis + np.array([0.0, 0.0, 0.34], dtype=np.float64)
    up = mid - pelvis
    un = float(np.linalg.norm(up))
    if un < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        up = up / un
    return mid + up * float(offset_m)


def apply_joint_limits(targets: SiteDict, rules: tuple[JointLimitRule, ...]) -> None:
    for rule in rules:
        a = get_site(targets, rule.root)
        b = get_site(targets, rule.joint)
        c = get_site(targets, rule.tip)
        if a is None or b is None or c is None:
            continue
        c2 = clamp_joint_angle(
            a, b, c, min_deg=rule.min_deg, max_deg=rule.max_deg
        )
        if rule.move_tip:
            set_site(targets, rule.tip, c2)


def apply_point_segment_clearance(
    targets: SiteDict,
    rules: tuple[PointSegmentRule, ...],
) -> None:
    for rule in rules:
        p = get_site(targets, rule.point)
        seg_pts = _segment_endpoints(targets, rule.seg)
        if p is None or seg_pts is None:
            continue
        sa, sb = seg_pts
        moved = push_point_outside_capsule(p, sa, sb, rule.seg.radius_m, rule.clearance_m)
        if rule.movable_point:
            set_site(targets, rule.point, moved)


def apply_point_point_clearance(
    targets: SiteDict,
    rules: tuple[PointPointRule, ...],
) -> None:
    for rule in rules:
        p = get_site(targets, rule.p)
        q = get_site(targets, rule.q)
        if p is None or q is None:
            continue
        p2, q2 = push_points_apart(
            p,
            q,
            rule.min_dist_m,
            move_p=rule.movable_p,
            move_q=rule.movable_q,
        )
        if rule.movable_p:
            set_site(targets, rule.p, p2)
        if rule.movable_q:
            set_site(targets, rule.q, q2)


def solve_two_bone_mid(
    anchor: np.ndarray,
    tip: np.ndarray,
    lab: float,
    lbc: float,
    bend_hint: np.ndarray,
) -> np.ndarray:
    """固定肩+腕，求肘位置满足两段骨长。"""
    anchor = np.asarray(anchor, dtype=np.float64)
    tip = np.asarray(tip, dtype=np.float64)
    sw = tip - anchor
    d = float(np.linalg.norm(sw))
    if d < 1e-9:
        hint = np.asarray(bend_hint, dtype=np.float64).reshape(3)
        hn = float(np.linalg.norm(hint))
        if hn < 1e-9:
            hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            hint = hint / hn
        return anchor + hint * float(lab)

    sw_u = sw / d
    reach = float(lab + lbc)
    if d >= reach:
        return anchor + sw_u * float(lab)

    a = (lab * lab - lbc * lbc + d * d) / (2.0 * d)
    h = math.sqrt(max(0.0, lab * lab - a * a))
    mid_on_line = anchor + sw_u * a

    hint = np.asarray(bend_hint, dtype=np.float64).reshape(3)
    hint = hint - sw_u * float(np.dot(hint, sw_u))
    hn = float(np.linalg.norm(hint))
    if hn < 1e-9:
        hint = _perpendicular_unit(sw_u)
    else:
        hint = hint / hn
    return mid_on_line + hint * h


def _torso_frame(
    targets: SiteDict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    pelvis = get_site(targets, "pelvis")
    if pelvis is None:
        return None
    ls = get_site(targets, "left_shoulder")
    rs = get_site(targets, "right_shoulder")
    if ls is not None and rs is not None:
        top = 0.5 * (ls + rs)
        lateral = ls - rs
    else:
        top = pelvis + np.array([0.0, 0.0, 0.34], dtype=np.float64)
        lateral = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    up = top - pelvis
    un = float(np.linalg.norm(up))
    if un < 1e-6:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    else:
        up = up / un
    ln = float(np.linalg.norm(lateral))
    if ln < 1e-6:
        lateral = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        lateral = lateral / ln
    return pelvis, top, lateral


def _arm_bend_hint(
    shoulder: np.ndarray,
    elbow: np.ndarray,
    *,
    lateral: np.ndarray,
    side_sign: float,
) -> np.ndarray:
    """side_sign: 左臂 +1，右臂 -1（相对肩线 lateral）。"""
    lat = np.asarray(lateral, dtype=np.float64) * float(side_sign)
    hint = np.cross(lat, shoulder - elbow)
    hn = float(np.linalg.norm(hint))
    if hn < 1e-9:
        hint = lat
    else:
        hint = hint / hn
    # 保留原肘的弯曲侧
    if np.dot(hint, elbow - shoulder) < 0.0:
        hint = -hint
    return hint


def _enforce_lateral_halfspace(
    p: np.ndarray,
    pelvis: np.ndarray,
    top: np.ndarray,
    lateral: np.ndarray,
    side_sign: float,
    margin_m: float,
) -> np.ndarray:
    """左/右臂点须在躯干中心线相应侧，至少 margin 横向偏移。"""
    q, _ = _closest_on_segment(p, pelvis, top)
    off = p - q
    lat_dist = float(side_sign) * float(np.dot(off, lateral))
    if lat_dist >= float(margin_m):
        return p.copy()
    lat_comp = lateral * float(np.dot(off, lateral))
    radial = off - lat_comp
    return q + lateral * (float(side_sign) * float(margin_m)) + radial


def _push_arm_chain_outside_torso(
    targets: SiteDict,
    shoulder_n: str,
    elbow_n: str,
    wrist_n: str,
    *,
    pelvis: np.ndarray,
    top: np.ndarray,
    torso_radius_m: float,
    torso_clearance_m: float,
    wrist_extra_radius_m: float,
) -> None:
    shoulder = get_site(targets, shoulder_n)
    elbow = get_site(targets, elbow_n)
    wrist = get_site(targets, wrist_n)
    if shoulder is None:
        return

    elbow_r = torso_radius_m + torso_clearance_m
    wrist_r = torso_radius_m + torso_clearance_m + wrist_extra_radius_m

    if elbow is not None:
        moved = push_point_outside_capsule(elbow, pelvis, top, elbow_r, 0.0)
        set_site(targets, elbow_n, moved)
        elbow = moved

    if wrist is not None:
        moved = push_point_outside_capsule(wrist, pelvis, top, wrist_r, 0.0)
        set_site(targets, wrist_n, moved)
        wrist = moved

    # 前臂/上臂段中点也推出躯干，避免「点在外、段穿胸」
    if elbow is not None and wrist is not None:
        mid = 0.5 * (elbow + wrist)
        moved_mid = push_point_outside_capsule(mid, pelvis, top, elbow_r, 0.0)
        if float(np.linalg.norm(moved_mid - mid)) > 1e-6:
            delta = moved_mid - mid
            set_site(targets, elbow_n, elbow + 0.5 * delta)
            set_site(targets, wrist_n, wrist + 0.5 * delta)

    if shoulder is not None and elbow is not None:
        mid = 0.5 * (shoulder + elbow)
        moved_mid = push_point_outside_capsule(mid, pelvis, top, elbow_r, 0.0)
        if float(np.linalg.norm(moved_mid - mid)) > 1e-6:
            delta = moved_mid - mid
            set_site(targets, elbow_n, get_site(targets, elbow_n) + delta)


def _push_segment_outside_capsule(
    p1: np.ndarray,
    p2: np.ndarray,
    pelvis: np.ndarray,
    top: np.ndarray,
    radius: float,
    clearance: float,
    *,
    n_samples: int = 9,
    n_iter: int = 6,
    move_start: bool = False,
    move_end: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """骨段推出躯干胶囊；默认只动末端（上臂动肘、前臂动腕）。"""
    a = np.asarray(p1, dtype=np.float64).copy()
    b = np.asarray(p2, dtype=np.float64).copy()
    need = float(radius) + float(clearance)
    ts = np.linspace(0.0, 1.0, int(n_samples))
    for _ in range(int(n_iter)):
        moved = False
        for t in ts:
            p = a * (1.0 - t) + b * t
            q, dist = _closest_on_segment(p, pelvis, top)
            if dist >= need:
                continue
            outward = p - q
            out_len = float(np.linalg.norm(outward))
            if out_len < 1e-6:
                outward = p - pelvis
                out_len = float(np.linalg.norm(outward))
            if out_len < 1e-6:
                outward = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                out_len = 1.0
            delta = outward * ((need - dist) / out_len)
            if move_start and t > 1e-6:
                a += delta * (1.0 - t)
            if move_end and t < 1.0 - 1e-6:
                b += delta * t
            moved = True
        if not moved:
            break
    return a, b


def _refit_arm_two_bone(
    targets: SiteDict,
    sh: str,
    el: str,
    wr: str,
    *,
    lateral: np.ndarray,
    side_sign: float,
    upper_arm_m: float,
    forearm_m: float,
) -> None:
    shoulder = get_site(targets, sh)
    wrist = get_site(targets, wr)
    elbow = get_site(targets, el)
    if shoulder is None or wrist is None:
        return
    hint = _arm_bend_hint(
        shoulder,
        elbow if elbow is not None else shoulder,
        lateral=lateral,
        side_sign=side_sign,
    )
    set_site(
        targets,
        el,
        solve_two_bone_mid(
            shoulder,
            wrist,
            float(upper_arm_m),
            float(forearm_m),
            hint,
        ),
    )


def apply_arms_torso_final(
    targets: SiteDict,
    *,
    upper_arm_m: float,
    forearm_m: float,
    torso_radius_m: float,
    torso_clearance_m: float,
    wrist_extra_radius_m: float = 0.05,
    lateral_elbow_margin_m: float = 0.08,
    lateral_wrist_margin_m: float = 0.12,
    iters: int = 5,
) -> None:
    """末段：臂段↔躯干硬避让 + 二节 IK 重拟合（避免 FABRIK 把腕拉回胸内）。"""
    del lateral_elbow_margin_m, lateral_wrist_margin_m
    frame = _torso_frame(targets)
    if frame is None:
        return
    pelvis, top, lateral = frame

    arms = (
        ("left_shoulder", "left_elbow", "left_wrist", 1.0),
        ("right_shoulder", "right_elbow", "right_wrist", -1.0),
    )
    elbow_r = torso_radius_m + torso_clearance_m + 0.02
    wrist_r = torso_radius_m + torso_clearance_m + wrist_extra_radius_m

    for _ in range(max(1, int(iters))):
        for sh, el, wr, sign in arms:
            shoulder = get_site(targets, sh)
            elbow = get_site(targets, el)
            wrist = get_site(targets, wr)
            if shoulder is None:
                continue

            if elbow is not None:
                _shoulder, elbow = _push_segment_outside_capsule(
                    shoulder,
                    elbow,
                    pelvis,
                    top,
                    elbow_r,
                    0.0,
                    move_start=False,
                    move_end=True,
                )
                set_site(targets, el, elbow)
            if elbow is not None and wrist is not None:
                _elbow, wrist = _push_segment_outside_capsule(
                    elbow,
                    wrist,
                    pelvis,
                    top,
                    wrist_r,
                    0.0,
                    move_start=False,
                    move_end=True,
                )
                set_site(targets, wr, wrist)
                _refit_arm_two_bone(
                    targets,
                    sh,
                    el,
                    wr,
                    lateral=lateral,
                    side_sign=sign,
                    upper_arm_m=upper_arm_m,
                    forearm_m=forearm_m,
                )

            elbow = get_site(targets, el)
            wrist = get_site(targets, wr)
            if elbow is not None:
                set_site(
                    targets,
                    el,
                    push_point_outside_capsule(elbow, pelvis, top, elbow_r, 0.0),
                )
            if wrist is not None:
                set_site(
                    targets,
                    wr,
                    push_point_outside_capsule(wrist, pelvis, top, wrist_r, 0.0),
                )
            _refit_arm_two_bone(
                targets,
                sh,
                el,
                wr,
                lateral=lateral,
                side_sign=sign,
                upper_arm_m=upper_arm_m,
                forearm_m=forearm_m,
            )

    for sh, el, wr, sign in arms:
        shoulder = get_site(targets, sh)
        elbow = get_site(targets, el)
        wrist = get_site(targets, wr)
        if shoulder is None or elbow is None or wrist is None:
            continue
        _elbow, wrist = _push_segment_outside_capsule(
            elbow,
            wrist,
            pelvis,
            top,
            wrist_r,
            0.03,
            move_start=False,
            move_end=True,
        )
        set_site(targets, wr, wrist)
        _refit_arm_two_bone(
            targets,
            sh,
            el,
            wr,
            lateral=lateral,
            side_sign=sign,
            upper_arm_m=upper_arm_m,
            forearm_m=forearm_m,
        )
        elbow = get_site(targets, el)
        if elbow is not None:
            _shoulder, elbow = _push_segment_outside_capsule(
                shoulder,
                elbow,
                pelvis,
                top,
                elbow_r,
                0.02,
                move_start=False,
                move_end=True,
            )
            set_site(targets, el, elbow)
            _refit_arm_two_bone(
                targets,
                sh,
                el,
                wr,
                lateral=lateral,
                side_sign=sign,
                upper_arm_m=upper_arm_m,
                forearm_m=forearm_m,
            )


def torso_penetration_depth(
    targets: SiteDict,
    *,
    torso_radius_m: float,
    torso_clearance_m: float,
    wrist_extra_radius_m: float = 0.0,
) -> float:
    """最大躯干胶囊穿透深度（m）。"""
    frame = _torso_frame(targets)
    if frame is None:
        return 0.0
    pelvis, top, _ = frame
    worst = 0.0
    checks = (
        ("left_elbow", torso_radius_m + torso_clearance_m),
        ("right_elbow", torso_radius_m + torso_clearance_m),
        ("left_wrist", torso_radius_m + torso_clearance_m + wrist_extra_radius_m),
        ("right_wrist", torso_radius_m + torso_clearance_m + wrist_extra_radius_m),
    )
    for name, need in checks:
        p = get_site(targets, name)
        if p is None:
            continue
        _, dist = _closest_on_segment(p, pelvis, top)
        worst = max(worst, need - dist)
    return max(0.0, worst)


def apply_wrist_torso_clearance(
    targets: SiteDict,
    *,
    torso_radius_m: float,
    torso_clearance_m: float,
) -> None:
    pelvis = get_site(targets, "pelvis")
    if pelvis is None:
        return
    ls = get_site(targets, "left_shoulder")
    rs = get_site(targets, "right_shoulder")
    if ls is not None and rs is not None:
        top = 0.5 * (ls + rs)
    else:
        top = pelvis + np.array([0.0, 0.0, 0.34], dtype=np.float64)
    for name in _WRIST_TORSO_SITES:
        p = get_site(targets, name)
        if p is None:
            continue
        moved = push_point_outside_capsule(
            p, pelvis, top, torso_radius_m, torso_clearance_m
        )
        set_site(targets, name, moved)


def apply_forearm_head_clearance(
    targets: SiteDict,
    *,
    head_radius_m: float,
    clearance_m: float,
    head_offset_m: float,
) -> None:
    head = head_proxy(targets, head_offset_m)
    if head is None:
        return
    for wrist in ("left_wrist", "right_wrist"):
        p = get_site(targets, wrist)
        if p is None:
            continue
        moved = push_point_outside_capsule(
            p, head, head, head_radius_m, clearance_m
        )
        set_site(targets, wrist, moved)


def max_clearance_violation(targets: SiteDict, min_dist_m: float = 0.03) -> float:
    """粗略：点-点 / 点-段 clearance 违反量（m）。"""
    worst = 0.0
    for rule in POINT_POINT_CLEARANCE_RULES:
        p = get_site(targets, rule.p)
        q = get_site(targets, rule.q)
        if p is None or q is None:
            continue
        d = float(np.linalg.norm(p - q))
        worst = max(worst, rule.min_dist_m - d)
    for rule in POINT_SEGMENT_CLEARANCE_RULES:
        p = get_site(targets, rule.point)
        seg_pts = _segment_endpoints(targets, rule.seg)
        if p is None or seg_pts is None:
            continue
        _, dist = _closest_on_segment(p, seg_pts[0], seg_pts[1])
        need = rule.seg.radius_m + rule.clearance_m
        worst = max(worst, need - dist)
    return max(0.0, worst)


def limit_frame_shift(
    cur: SiteDict,
    prev: SiteDict | None,
    max_shift_m: float,
) -> None:
    """VelocityLimit 近似：相对上一帧单点最大位移。"""
    if prev is None or max_shift_m <= 0.0:
        return
    for name in SITE_NAMES:
        p = get_site(cur, name)
        q = get_site(prev, name)
        if p is None or q is None:
            continue
        d = p - q
        n = float(np.linalg.norm(d))
        if n <= max_shift_m or n < 1e-9:
            continue
        set_site(cur, name, q + d * (max_shift_m / n))
