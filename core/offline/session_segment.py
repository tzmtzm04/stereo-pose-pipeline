"""离线 Anipose 流水线前的 session 帧段切分（jsonl 行序，0-based）。"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class OfflineSessionSegment:
    """闭区间 [start_index, end_index]（含首尾）。"""

    start_index: int = 100
    end_index: int = 900

    def __post_init__(self) -> None:
        s, e = int(self.start_index), int(self.end_index)
        if s < 0:
            raise ValueError(f"start_index 须 >= 0， got {s}")
        if e < s:
            raise ValueError(f"end_index ({e}) 须 >= start_index ({s})")

    @property
    def length(self) -> int:
        return int(self.end_index) - int(self.start_index) + 1

    def slice(self, n_frames: int) -> slice:
        n = int(n_frames)
        s = int(self.start_index)
        e = min(int(self.end_index), n - 1)
        if s >= n:
            raise ValueError(
                f"segment start_index={s} 超出帧数 {n}"
            )
        return slice(s, e + 1)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# 显式 --segment 时使用的参考段（入画后、出画前；session 20260713_174204 标定）。
# 默认流水线不切段（全程）；勿在 resolve 之外隐式套用。
DEFAULT_OFFLINE_SEGMENT = OfflineSessionSegment(start_index=100, end_index=900)


def resolve_offline_segment(
    *,
    segment: bool = False,
    segment_start: int | None = None,
    segment_end: int | None = None,
) -> OfflineSessionSegment | None:
    """默认 None（全程）；仅 --segment 或 --segment-start/end 时切段。"""
    if segment_start is not None or segment_end is not None:
        if segment_start is None or segment_end is None:
            raise ValueError("--segment-start 与 --segment-end 须同时指定")
        return OfflineSessionSegment(
            start_index=int(segment_start),
            end_index=int(segment_end),
        )
    if segment:
        return DEFAULT_OFFLINE_SEGMENT
    return None


def add_offline_segment_cli(ap: argparse.ArgumentParser) -> None:
    ap.add_argument(
        "--segment",
        action="store_true",
        help=f"切参考段 index {DEFAULT_OFFLINE_SEGMENT.start_index}–"
        f"{DEFAULT_OFFLINE_SEGMENT.end_index}（默认全程）",
    )
    ap.add_argument(
        "--segment-start",
        type=int,
        default=None,
        help="自定义段起始帧（0-based，含；须与 --segment-end 同用）",
    )
    ap.add_argument(
        "--segment-end",
        type=int,
        default=None,
        help="自定义段结束帧（0-based，含；须与 --segment-start 同用）",
    )
    ap.add_argument(
        "--no-segment",
        action="store_true",
        help="[兼容] 显式全程（默认已是全程）",
    )


def parse_offline_segment_cli(args: argparse.Namespace) -> OfflineSessionSegment | None:
    if bool(getattr(args, "no_segment", False)):
        return None
    try:
        return resolve_offline_segment(
            segment=bool(getattr(args, "segment", False)),
            segment_start=getattr(args, "segment_start", None),
            segment_end=getattr(args, "segment_end", None),
        )
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


def format_offline_segment(segment: OfflineSessionSegment | None) -> str:
    if segment is None:
        return "off"
    return str(segment.to_dict())


def apply_segment_pose_stack(
    points: np.ndarray,
    scores: np.ndarray,
    rows: list[dict],
    segment: OfflineSessionSegment | None,
) -> tuple[np.ndarray, np.ndarray, list[dict], OfflineSessionSegment | None]:
    """对 load_pose2d 结果切分；segment=None 则原样返回。"""
    if segment is None:
        return points, scores, rows, None
    sl = segment.slice(points.shape[1])
    return (
        points[:, sl],
        scores[:, sl],
        rows[sl],
        segment,
    )
