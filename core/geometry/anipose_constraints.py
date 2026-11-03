"""WB133 / COCO-WholeBody 骨段 constraints（aniposelib optim_points 用）。"""

from __future__ import annotations

from core.offline.two_stage.coco_wholebody_layout import (
    BODY_FEET_SKELETON,
    NUM_KEYPOINTS_BODY_FEET,
)

# body+feet 23 点骨段（与 coco_wholebody_layout.BODY_FEET_SKELETON 一致）
BODY_FEET_BONE_CONSTRAINTS: tuple[tuple[int, int], ...] = tuple(BODY_FEET_SKELETON)

# 腕（肘–腕）与踝（膝–踝）弱骨长；local/global idx 均为 COCO body_feet 0–22
BODY_FEET_WRIST_ANKLE_WEAK_BONES: tuple[tuple[int, int], ...] = (
    (7, 9),
    (8, 10),
    (13, 15),
    (14, 16),
)

# 仅 body 17 点（不含脚）
BODY_BONE_CONSTRAINTS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

# 躯干+四肢 12 点（COCO 5–16，无头/耳/脚）；constraints 为 optim 数组局部下标
BODY_LIMBS_JOINT_INDICES: tuple[int, ...] = tuple(range(5, 17))
BODY_LIMBS_BONE_CONSTRAINTS: tuple[tuple[int, int], ...] = (
    (0, 1),   # 肩
    (0, 2),
    (2, 4),   # 左臂
    (1, 3),
    (3, 5),   # 右臂
    (0, 6),
    (1, 7),   # 肩→髋
    (6, 7),   # 髋
    (6, 8),
    (8, 10),  # 左腿
    (7, 9),
    (9, 11),  # 右腿
)


def weak_constraints_for_joint_set(name: str) -> tuple[tuple[int, int], ...]:
    key = str(name).strip().lower().replace("-", "_")
    if key in ("body_feet", "body+feet", "23", "bf"):
        return BODY_FEET_WRIST_ANKLE_WEAK_BONES
    return ()


def constraints_for_joint_set(name: str) -> list[list[int]]:
    """返回 aniposelib `constraints=[[a,b],...]` 格式。"""
    key = str(name).strip().lower().replace("-", "_")
    if key in ("body_feet", "body+feet", "23", "bf"):
        pairs = BODY_FEET_BONE_CONSTRAINTS
    elif key in ("body", "17", "coco17"):
        pairs = BODY_BONE_CONSTRAINTS
    elif key in ("body_limbs", "limbs", "no_head", "body12"):
        pairs = BODY_LIMBS_BONE_CONSTRAINTS
    else:
        raise ValueError(
            f"未知 joint set: {name!r}（body_feet | body | body_limbs）"
        )
    return [[int(a), int(b)] for a, b in pairs]


def default_joint_indices(name: str) -> tuple[int, ...]:
    key = str(name).strip().lower().replace("-", "_")
    if key in ("body_feet", "body+feet", "23", "bf"):
        return tuple(range(NUM_KEYPOINTS_BODY_FEET))
    if key in ("body", "17", "coco17"):
        return tuple(range(17))
    if key in ("body_limbs", "limbs", "no_head", "body12"):
        return BODY_LIMBS_JOINT_INDICES
    raise ValueError(f"未知 joint set: {name!r}")
