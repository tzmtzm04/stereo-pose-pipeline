"""MuJoCo IK site 顺序与骨架连线（供 3D preview）。"""

from __future__ import annotations

from core.mujoco_ik.layout import SITE_NAMES

# IK 目标 site（13 点，与 layout 一致，含腕踝）
MUJOCO_IK_TARGET_SITE_NAMES: tuple[str, ...] = SITE_NAMES

# 腕踝已并入 IK 目标，preview 不再单独列出
MUJOCO_IK_EXTREMITY_SITE_NAMES: tuple[str, ...] = ()

MUJOCO_IK_PREVIEW_SITE_NAMES: tuple[str, ...] = MUJOCO_IK_TARGET_SITE_NAMES

# 兼容旧名
MUJOCO_IK_SITE_NAMES: tuple[str, ...] = MUJOCO_IK_PREVIEW_SITE_NAMES

# 0=pelvis … 12=右踝（与 layout.IK_SITE_SPECS 顺序一致）
MUJOCO_IK_SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 4),
    (0, 7),
    (0, 10),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6),
    (7, 8),
    (8, 9),
    (10, 11),
    (11, 12),
    (1, 4),
    (7, 10),
]

MUJOCO_IK_CENTER_INDICES: tuple[int, ...] = (0, 7, 10)

MUJOCO_IK_LAYOUT_LABEL = "MuJoCo IK (13 sites incl. wrist/ankle)"


def site_index(name: str) -> int:
    return MUJOCO_IK_PREVIEW_SITE_NAMES.index(name)
