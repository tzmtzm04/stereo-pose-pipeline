"""OpenSim/H36M+ surface marker 拓扑（BioModule v1 dimensionless_marker，预览用 M1–M19）。"""

from __future__ import annotations

# UTSA build_aligned_dataset.py 官方文档 marker 数
MARKER_COUNT = 19
# v1 模型 dimensionless_marker 仍输出 M1–M20（60 维），预览只取前 19 个
MARKER_MODEL_COUNT = 20

# build_aligned_dataset.py: M1=pelvis … M16–M19=R leg
MARKER_NAMES: tuple[str, ...] = (
    "M1_pelvis",
    "M2_thorax",
    "M3_neck",
    "M4_L_shoulder",
    "M5_L_elbow",
    "M6_L_wrist",
    "M7_L_hand",
    "M8_R_shoulder",
    "M9_R_elbow",
    "M10_R_wrist",
    "M11_R_hand",
    "M12_L_hip",
    "M13_L_knee",
    "M14_L_ankle",
    "M15_L_foot",
    "M16_R_hip",
    "M17_R_knee",
    "M18_R_ankle",
    "M19_R_foot",
)

# 右窗点旁标注（中文；OpenCV 默认字体可能显示为方框，见 skeleton3d_view 英文回退）
MARKER_LABELS: tuple[str, ...] = (
    "骨盆",
    "胸",
    "颈",
    "左肩",
    "左肘",
    "左腕",
    "左手",
    "右肩",
    "右肘",
    "右腕",
    "右手",
    "左髋",
    "左膝",
    "左踝",
    "左脚",
    "右髋",
    "右膝",
    "右踝",
    "右脚",
)

MARKER_LABELS_ASCII: tuple[str, ...] = tuple(
    n.split("_", 1)[1].replace("_", "-") if "_" in n else n for n in MARKER_NAMES
)

# 0-indexed 连线（M1 骨盆为根；预览推断，非 OpenSim 官方边表）
MARKER_SKELETON_EDGES: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (1, 3),
    (3, 4),
    (4, 5),
    (5, 6),
    (1, 7),
    (7, 8),
    (8, 9),
    (9, 10),
    (0, 11),
    (11, 12),
    (12, 13),
    (13, 14),
    (0, 15),
    (15, 16),
    (16, 17),
    (17, 18),
]
