"""Session 结束后的 L3 判段（读 session_pose.jsonl，不依赖内存 PoseHistory）。"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from domain.l3.plugins.mock_detector import MockBoundarySpec, default_mock_segments

if TYPE_CHECKING:
    from domain.l3.runtime import L3Runtime

logger = logging.getLogger(__name__)


def max_capture_seq_from_pose_log(path: Path) -> int:
    """扫描 session_pose.jsonl 得最大 capture_seq（文件不存在则 -1）。"""
    p = Path(path)
    if not p.is_file():
        return -1
    max_seq = -1
    with p.open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            seq = int(row.get("capture_seq", -1))
            if seq > max_seq:
                max_seq = seq
    return max_seq


def run_mock_session_post_process(
    l3: "L3Runtime",
    *,
    pose_path: Path,
    segments: tuple[MockBoundarySpec, ...] | None = None,
) -> list[tuple[int, int, Optional[str]]]:
    """
    退出时登记 mock 段 → manifest post_process。
    仅当 max(capture_seq) >= segment.end 时登记该段。
    """
    specs = segments if segments is not None else default_mock_segments()
    max_seq = max_capture_seq_from_pose_log(pose_path)
    registered: list[tuple[int, int, Optional[str]]] = []
    if max_seq < 0:
        logger.warning("session post_process: no pose rows in %s", pose_path)
        return registered

    for spec in specs:
        if max_seq < spec.end_capture_seq:
            logger.info(
                "session post_process: skip segment %d..%s (max_seq=%d)",
                spec.start_capture_seq,
                spec.end_capture_seq,
                max_seq,
            )
            continue
        l3.register_action_segment(
            spec.start_capture_seq,
            spec.end_capture_seq,
            label=spec.label,
        )
        registered.append(
            (spec.start_capture_seq, spec.end_capture_seq, spec.label)
        )
        logger.info(
            "session post_process: registered %d..%d label=%s (max_seq=%d)",
            spec.start_capture_seq,
            spec.end_capture_seq,
            spec.label,
            max_seq,
        )
    return registered
