"""Pose2Sim 方案 B：外部 TRC 接入 filtering / IK。"""

from core.pose2sim.coco17_layout import COCO17_MARKER_NAMES, COCO17_P2S_MARKERS
from core.pose2sim.trc_export import TrcExportStats, export_session_to_pose2sim_trial

__all__ = [
    "COCO17_MARKER_NAMES",
    "COCO17_P2S_MARKERS",
    "TrcExportStats",
    "export_session_to_pose2sim_trial",
]
