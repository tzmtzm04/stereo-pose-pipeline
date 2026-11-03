"""wb133_ik_humanoid 自碰撞 geom 组与 mink CollisionAvoidanceLimit 配置。

启用步骤：
1. MJCF geom 需 contype/conaffinity≠0（见 models/mujoco/wb133_ik_humanoid.xml class=collision）
2. MinkIKConfig.enable_collision 默认 True；CLI 用 --no-collision 关闭
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mujoco


# --- geom 名（与 MJCF name= 一致）---
PELVIS = "pelvis_geom"
TORSO = "torso_geom"
HEAD = "head_geom"
LEFT_CLAVICLE = "left_clavicle_geom"
LEFT_UPPER_ARM = "left_upper_arm_geom"
LEFT_FOREARM = "left_forearm_geom"
RIGHT_CLAVICLE = "right_clavicle_geom"
RIGHT_UPPER_ARM = "right_upper_arm_geom"
RIGHT_FOREARM = "right_forearm_geom"
LEFT_THIGH = "left_thigh_geom"
LEFT_SHIN = "left_shin_geom"
LEFT_FOOT = "left_foot_geom"
RIGHT_THIGH = "right_thigh_geom"
RIGHT_SHIN = "right_shin_geom"
RIGHT_FOOT = "right_foot_geom"

ALL_COLLISION_GEOMS: tuple[str, ...] = (
    PELVIS,
    TORSO,
    HEAD,
    LEFT_CLAVICLE,
    LEFT_UPPER_ARM,
    LEFT_FOREARM,
    RIGHT_CLAVICLE,
    RIGHT_UPPER_ARM,
    RIGHT_FOREARM,
    LEFT_THIGH,
    LEFT_SHIN,
    LEFT_FOOT,
    RIGHT_THIGH,
    RIGHT_SHIN,
    RIGHT_FOOT,
)

# geom 组（mink 要求：pair = (group_a, group_b)，组内笛卡尔积）
PELVIS_G = (PELVIS,)
TORSO_G = (TORSO,)
HEAD_G = (HEAD,)
LEFT_ARM_G = (LEFT_CLAVICLE, LEFT_UPPER_ARM, LEFT_FOREARM)
RIGHT_ARM_G = (RIGHT_CLAVICLE, RIGHT_UPPER_ARM, RIGHT_FOREARM)
LEFT_LEG_G = (LEFT_THIGH, LEFT_SHIN, LEFT_FOOT)
RIGHT_LEG_G = (RIGHT_THIGH, RIGHT_SHIN, RIGHT_FOOT)
TRUNK_G = (PELVIS, TORSO, HEAD)

# 不列相邻连杆（父–子 body 在 mink 里也会部分过滤，仍建议显式省略）：
# pelvis–torso, torso–head, clavicle–upper_arm–forearm, thigh–shin–foot

WB133_IK_COLLISION_PAIRS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    # 手臂 ↔ 躯干
    (LEFT_ARM_G, TRUNK_G),
    (RIGHT_ARM_G, TRUNK_G),
    # 左右臂交叉
    (LEFT_ARM_G, RIGHT_ARM_G),
    # 手臂 ↔ 对侧腿（抱腿、手过中线）
    (LEFT_ARM_G, RIGHT_LEG_G),
    (RIGHT_ARM_G, LEFT_LEG_G),
    # 同侧臂 ↔ 同侧腿（手贴大腿外侧仍可能接近）
    ((LEFT_FOREARM,), LEFT_LEG_G),
    ((RIGHT_FOREARM,), RIGHT_LEG_G),
    # 双腿剪叉 / 并腿
    (LEFT_LEG_G, RIGHT_LEG_G),
    # 头 ↔ 前臂（护头、摸头）
    ((LEFT_FOREARM, RIGHT_FOREARM), HEAD_G),
)


@dataclass(frozen=True)
class CollisionAvoidanceConfig:
    """mink CollisionAvoidanceLimit 参数。"""

    gain: float = 0.85
    minimum_distance_m: float = 0.03
    detection_distance_m: float = 0.06
    bound_relaxation: float = 0.0
    broadphase: bool = True


DEFAULT_COLLISION_CFG = CollisionAvoidanceConfig()


def describe_collision_pairs() -> str:
    lines = ["WB133 IK 自碰撞 geom 对（节选）:"]
    for i, (a, b) in enumerate(WB133_IK_COLLISION_PAIRS, 1):
        lines.append(f"  {i}. {list(a)} × {list(b)}")
    return "\n".join(lines)


def build_collision_avoidance_limit(
    model: mujoco.MjModel,
    cfg: CollisionAvoidanceConfig | None = None,
):
    """构造 mink CollisionAvoidanceLimit；geom 未开碰撞或未命名时抛错。"""
    import mujoco
    import mink

    cfg = cfg or DEFAULT_COLLISION_CFG
    missing: list[str] = []
    for name in ALL_COLLISION_GEOMS:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name) < 0:
            missing.append(name)
    if missing:
        raise ValueError(f"MJCF 缺少 collision geom: {missing}")

    inactive = [
        n
        for n in ALL_COLLISION_GEOMS
        if model.geom_contype[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)] == 0
    ]
    if inactive:
        raise ValueError(
            f"geom contype=0，mj_geomDistance 无效: {inactive}。"
            "请给 MJCF geom 设 class=collision 或 contype/conaffinity=1"
        )

    return mink.CollisionAvoidanceLimit(
        model,
        geom_pairs=list(WB133_IK_COLLISION_PAIRS),
        gain=float(cfg.gain),
        minimum_distance_from_collisions=float(cfg.minimum_distance_m),
        collision_detection_distance=float(cfg.detection_distance_m),
        bound_relaxation=float(cfg.bound_relaxation),
        broadphase=bool(cfg.broadphase),
    )
