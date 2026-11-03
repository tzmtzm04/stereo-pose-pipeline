"""Path 6 MSK + mink IK：冻结未观测关节、根锚 site、骨盆/目标投影默认。"""

from __future__ import annotations

# Tier A+B：无 marker / 应对齐 OpenSim 锁定语义
# Walker 膝辅助关节不在此冻结 —— 由 MJCF equality + EqualityConstraintTask 耦合 knee_angle
PATH6_FROZEN_UNOBSERVED_JOINT_NAMES: tuple[str, ...] = (
    "pro_sup_r",
    "pro_sup_l",
    "subtalar_angle_r",
    "subtalar_angle_l",
    "mtp_angle_r",
    "mtp_angle_l",
    "neck_flexion",
    "neck_bending",
    "neck_rotation",
    "L5_S1_Flex_Ext",
    "L5_S1_Lat_Bending",
    "L5_S1_axial_rotation",
    "arm_rot_r",
    "arm_rot_l",
)

# 骨盆旋转 DOF 冗余大；对齐 wb133「少动根」思路（保留 tx/ty/tz 贴 hip marker）
PATH6_FROZEN_PELVIS_JOINT_NAMES: tuple[str, ...] = (
    "pelvis_tilt",
    "pelvis_rotation",
)

PATH6_FROZEN_JOINT_NAMES: tuple[str, ...] = (
    PATH6_FROZEN_UNOBSERVED_JOINT_NAMES + PATH6_FROZEN_PELVIS_JOINT_NAMES
)

PATH6_ROOT_ANCHOR_SITES: tuple[str, ...] = ("LHip_marker", "RHip_marker")

DEFAULT_PATH6_MINK_POSTURE_COST = 0.02
DEFAULT_PATH6_MINK_MAX_STEPS = 50
# 对齐 OpenSim constraint_weight=Inf（Walker 膝 joint equality）
DEFAULT_PATH6_EQUALITY_COST = 100.0
