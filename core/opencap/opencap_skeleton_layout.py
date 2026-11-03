"""OpenCap 增广 TRC 解剖 marker 骨架连线（按 marker 名解析）。"""

from __future__ import annotations

from dataclasses import dataclass

from core.opencap.opencap20_layout import OPENCAP20_MARKER_NAMES

# OpenPose-20（TRC 前 20 列，与 OPENCAP20_MARKER_NAMES 一致）
_OPENCAP20_EDGES: list[tuple[str, str]] = [
    ("Neck", "RShoulder"),
    ("Neck", "LShoulder"),
    ("RShoulder", "LShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
    ("RShoulder", "RHip"),
    ("LShoulder", "LHip"),
    ("RHip", "LHip"),
    ("RHip", "midHip"),
    ("LHip", "midHip"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("RAnkle", "RHeel"),
    ("RAnkle", "RBigToe"),
    ("RAnkle", "RSmallToe"),
    ("LAnkle", "LHeel"),
    ("LAnkle", "LBigToe"),
    ("LAnkle", "LSmallToe"),
]

# LSTM 解剖 marker（OpenCap / LaiUhlrich study 命名）
_ANATOMICAL_EDGES: list[tuple[str, str]] = [
    # 骨盆
    ("r.ASIS_study", "L.ASIS_study"),
    ("r.PSIS_study", "L.PSIS_study"),
    ("r.ASIS_study", "r.PSIS_study"),
    ("L.ASIS_study", "L.PSIS_study"),
    ("RHJC_study", "LHJC_study"),
    ("RHJC_study", "r.ASIS_study"),
    ("LHJC_study", "L.ASIS_study"),
    ("RHJC_study", "midHip"),
    ("LHJC_study", "midHip"),
    # 躯干
    ("C7_study", "Neck"),
    ("C7_study", "r_shoulder_study"),
    ("C7_study", "L_shoulder_study"),
    ("r_shoulder_study", "RShoulder"),
    ("L_shoulder_study", "LShoulder"),
    # 右下肢
    ("RHJC_study", "r_thigh1_study"),
    ("r_thigh1_study", "r_thigh2_study"),
    ("r_thigh2_study", "r_thigh3_study"),
    ("r_thigh3_study", "r_knee_study"),
    ("r_knee_study", "r_mknee_study"),
    ("r_mknee_study", "r_ankle_study"),
    ("r_ankle_study", "r_mankle_study"),
    ("r_ankle_study", "r_calc_study"),
    ("r_calc_study", "r_toe_study"),
    ("r_toe_study", "r_5meta_study"),
    ("r_sh1_study", "r_sh2_study"),
    ("r_sh2_study", "r_sh3_study"),
    ("r_sh3_study", "r_ankle_study"),
    # 左下肢
    ("LHJC_study", "L_thigh1_study"),
    ("L_thigh1_study", "L_thigh2_study"),
    ("L_thigh2_study", "L_thigh3_study"),
    ("L_thigh3_study", "L_knee_study"),
    ("L_knee_study", "L_mknee_study"),
    ("L_mknee_study", "L_ankle_study"),
    ("L_ankle_study", "L_mankle_study"),
    ("L_ankle_study", "L_calc_study"),
    ("L_calc_study", "L_toe_study"),
    ("L_toe_study", "L_5meta_study"),
    ("L_sh1_study", "L_sh2_study"),
    ("L_sh2_study", "L_sh3_study"),
    ("L_sh3_study", "L_ankle_study"),
    # 上肢
    ("r_lelbow_study", "r_melbow_study"),
    ("r_lwrist_study", "r_mwrist_study"),
    ("r_melbow_study", "r_shoulder_study"),
    ("r_mwrist_study", "r_shoulder_study"),
    ("L_lelbow_study", "L_melbow_study"),
    ("L_lwrist_study", "L_mwrist_study"),
    ("L_melbow_study", "L_shoulder_study"),
    ("L_mwrist_study", "L_shoulder_study"),
]

OPENCAP_AUGMENTED_EDGES: list[tuple[str, str]] = _OPENCAP20_EDGES + _ANATOMICAL_EDGES

LAYOUT_OPENCAP_AUG = "opencap_aug"
# OpenCap-20 在增广 TRC 中的骨盆中心索引
OPENCAP_CENTER_INDICES = (5, 3, 4)  # midHip, RHip, LHip


@dataclass(frozen=True)
class OpencapSkeletonSpec:
    edges: list[tuple[int, int]]
    label: str
    num_markers: int


def resolve_opencap_skeleton_edges(marker_names: list[str]) -> OpencapSkeletonSpec:
    name_to_i = {n: i for i, n in enumerate(marker_names)}
    edges: list[tuple[int, int]] = []
    for a, b in OPENCAP_AUGMENTED_EDGES:
        if a in name_to_i and b in name_to_i:
            edges.append((name_to_i[a], name_to_i[b]))
    has_anat = any(n.endswith("_study") for n in marker_names)
    label = "OpenCap-63" if has_anat and len(marker_names) >= 40 else "OpenCap-20"
    return OpencapSkeletonSpec(
        edges=edges,
        label=label,
        num_markers=len(marker_names),
    )


def is_opencap_input_marker(name: str) -> bool:
    return name in OPENCAP20_MARKER_NAMES


def joint_style_opencap_aug(
    joint_idx: int,
    marker_names: list[str],
    *,
    dimmed: bool = False,
) -> tuple[int, tuple[int, int, int]]:
    """输入 20 点绿色，解剖 marker 橙色小点。"""
    if joint_idx < 0 or joint_idx >= len(marker_names):
        return 4, (120, 120, 140)
    name = marker_names[joint_idx]
    if is_opencap_input_marker(name):
        return 5, (0, 150, 200) if dimmed else (0, 220, 120)
    if name == "midHip":
        return 6, (0, 180, 255) if dimmed else (0, 220, 255)
    if name.endswith("_study"):
        return 3, (80, 140, 220) if dimmed else (100, 180, 255)
    return 4, (120, 120, 160) if dimmed else (180, 180, 200)
