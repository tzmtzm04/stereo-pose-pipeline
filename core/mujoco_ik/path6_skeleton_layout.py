"""Path 6 MSK MJCF marker site 顺序与 3D 骨架连线。"""

from __future__ import annotations

from core.mujoco_ik.path6_layout import marker_to_site_name

# marker_map 顺序 → site 名
PATH6_MARKER_ORDER: tuple[str, ...] = (
    "Nose",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
)

PATH6_PREVIEW_SITE_NAMES: tuple[str, ...] = tuple(
    marker_to_site_name(m) for m in PATH6_MARKER_ORDER
)

# 索引：0=Nose, 1-3=右臂, 4-6=左臂, 7-9=右腿, 10-12=左腿
PATH6_SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (0, 4),
    (1, 2),
    (2, 3),
    (4, 5),
    (5, 6),
    (7, 8),
    (8, 9),
    (10, 11),
    (11, 12),
    (7, 10),
    (1, 4),
]

PATH6_CENTER_INDICES: tuple[int, ...] = (7, 10)

PATH6_LAYOUT_LABEL = "Path6 MSK IK (13 markers)"

PATH6_KNEE_JOINT_NAMES: tuple[str, str] = ("knee_angle_l", "knee_angle_r")
