"""COCO-WholeBody 133 关键点布局（mmpose / ViTPose++ dataset_idx=5）。"""

from __future__ import annotations

# COCO-WholeBody v1.0：17 body + 6 feet + 68 face + 42 hand
NUM_KEYPOINTS_WHOLEBODY = 133
NUM_KEYPOINTS_BODY = 17
NUM_KEYPOINTS_FEET = 6
NUM_KEYPOINTS_BODY_FEET = NUM_KEYPOINTS_BODY + NUM_KEYPOINTS_FEET  # 23

# body 0-16 与标准 COCO-17 索引一致
BODY_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

FEET_KEYPOINT_NAMES: tuple[str, ...] = (
    "left_big_toe",
    "left_small_toe",
    "left_heel",
    "right_big_toe",
    "right_small_toe",
    "right_heel",
)

BODY_FEET_KEYPOINT_NAMES: tuple[str, ...] = BODY_KEYPOINT_NAMES + FEET_KEYPOINT_NAMES

# body+feet 中脚点索引（17–22，预览可隐藏）
FEET_JOINT_INDICES: tuple[int, ...] = tuple(
    range(NUM_KEYPOINTS_BODY, NUM_KEYPOINTS_BODY_FEET)
)

# body+feet 骨架（预览用）
BODY_FEET_SKELETON: list[tuple[int, int]] = [
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
    (15, 17),
    (15, 18),
    (15, 19),
    (16, 20),
    (16, 21),
    (16, 22),
]

# mmpose coco_wholebody skeleton_info（body+feet+hands）+ 腕→手掌根
WHOLEBODY_SKELETON_EDGES: list[tuple[int, int]] = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
    (15, 17),
    (15, 18),
    (15, 19),
    (16, 20),
    (16, 21),
    (16, 22),
    (91, 92),
    (92, 93),
    (93, 94),
    (94, 95),
    (91, 96),
    (96, 97),
    (97, 98),
    (98, 99),
    (91, 100),
    (100, 101),
    (101, 102),
    (102, 103),
    (91, 104),
    (104, 105),
    (105, 106),
    (106, 107),
    (91, 108),
    (108, 109),
    (109, 110),
    (110, 111),
    (112, 113),
    (113, 114),
    (114, 115),
    (115, 116),
    (112, 117),
    (117, 118),
    (118, 119),
    (119, 120),
    (112, 121),
    (121, 122),
    (122, 123),
    (123, 124),
    (112, 125),
    (125, 126),
    (126, 127),
    (127, 128),
    (112, 129),
    (129, 130),
    (130, 131),
    (131, 132),
    (9, 91),
    (10, 112),
]

WHOLEBODY_FACE_IDX_START = 23
WHOLEBODY_FACE_IDX_END = 91
WHOLEBODY_HAND_LEFT_IDX_START = 91
WHOLEBODY_HAND_RIGHT_IDX_START = 112
WHOLEBODY_HAND_IDX_END = NUM_KEYPOINTS_WHOLEBODY  # 91–132，不含腕 9/10


def skip_wholebody_hand_triangulation(
    joint_index: int,
    joint_count: int,
    *,
    enabled: bool = True,
) -> bool:
    """WholeBody 133 点布局下跳过左右手（91–132）三角化。"""
    if not enabled or joint_count < NUM_KEYPOINTS_WHOLEBODY:
        return False
    return WHOLEBODY_HAND_LEFT_IDX_START <= int(joint_index) < WHOLEBODY_HAND_IDX_END

WHOLEBODY_IDX_TO_OPENCAP_FOOT: dict[int, str] = {
    17: "LBigToe",
    18: "LSmallToe",
    19: "LHeel",
    20: "RBigToe",
    21: "RSmallToe",
    22: "RHeel",
}
