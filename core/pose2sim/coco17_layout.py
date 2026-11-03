"""COCO-17 → Pose2Sim Body (COCO_17) marker 顺序。"""

from __future__ import annotations

# Pose2Sim skeletons.COCO_17 RenderTree 顺序（与官方 triangulation make_trc 一致）
COCO17_P2S_MARKERS: tuple[tuple[str, int], ...] = (
    ("RHip", 12),
    ("RKnee", 14),
    ("RAnkle", 16),
    ("LHip", 11),
    ("LKnee", 13),
    ("LAnkle", 15),
    ("Nose", 0),
    ("RShoulder", 6),
    ("RElbow", 8),
    ("RWrist", 10),
    ("LShoulder", 5),
    ("LElbow", 7),
    ("LWrist", 9),
)

COCO17_MARKER_NAMES: tuple[str, ...] = tuple(n for n, _ in COCO17_P2S_MARKERS)
