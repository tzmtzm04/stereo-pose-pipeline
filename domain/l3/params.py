"""L3 → L1 运行与保存参数。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from contracts.nv12_layout import raw_ring_memory_mb_for_seconds
from contracts.segment import Segment, load_segments_file, segment_from_single
from domain.l3.config import (
    CLIP_OUTPUT_ROOT,
    DEFAULT_BITRATE,
    DEFAULT_IFRAME_INTERVAL,
    HISTORY_SECONDS,
    RAW_RING_MAX_MEMORY_MB,
    RAW_RING_SECONDS,
    history_max_samples,
)
from domain.registry.clip_registry import ClipRegistry


@dataclass
class L3Params:
    raw_ring_seconds: float = RAW_RING_SECONDS
    raw_ring_memory_mb: int = RAW_RING_MAX_MEMORY_MB
    enc_bitrate: int = DEFAULT_BITRATE
    enc_iframe_interval: int = DEFAULT_IFRAME_INTERVAL
    history_seconds: float = HISTORY_SECONDS

    clip_start: Optional[int] = None
    clip_end: Optional[int] = None
    clip_dir: Optional[Path] = None
    clips: Optional[tuple[Segment, ...]] = None
    clip_registry: Optional[ClipRegistry] = None

    def clip_specs(self) -> list[Segment]:
        if self.clip_registry is not None:
            return self.clip_registry.segments()
        if self.clips:
            return list(self.clips)
        if self.clip_start is not None and self.clip_end is not None:
            return segment_from_single(self.clip_start, self.clip_end)
        return []

    def history_max_samples(self, fps: int) -> int:
        return history_max_samples(fps)

    def append_clip_segment(
        self,
        start_capture_seq: int,
        end_capture_seq: int,
        *,
        label: Optional[str] = None,
    ) -> int:
        if self.clip_registry is None:
            raise RuntimeError("clip_registry not enabled (need --clip-dir)")
        return self.clip_registry.append_segment(
            start_capture_seq, end_capture_seq, label=label
        )

    def append_clip_marker(
        self,
        capture_seq: int,
        *,
        label: Optional[str] = None,
        kind: str = "marker",
    ) -> int:
        if self.clip_registry is None:
            raise RuntimeError("clip_registry not enabled (need --clip-dir)")
        return self.clip_registry.append_marker(
            capture_seq, label=label, kind=kind
        )

    def status_line(
        self,
        *,
        width: int | None = None,
        height: int | None = None,
        fps: int | None = None,
    ) -> str:
        specs = self.clip_specs()
        n = len(specs)
        if n == 0:
            clip_txt = "未设置"
        elif n == 1:
            s = specs[0]
            clip_txt = f"{s.start_capture_seq}..{s.end_capture_seq}"
        elif self.clip_registry is not None:
            clip_txt = f"registry={n}段(依次登记)"
        else:
            clip_txt = f"{n}段 {specs[0].start_capture_seq}..{specs[-1].end_capture_seq}"

        return (
            f"L3→L1: session_recording=on "
            f"iframe={self.enc_iframe_interval} history={self.history_seconds}s "
            f"clip={clip_txt}"
        )


def _resolve_clip_dir(
    *,
    clip_dir: Optional[Path],
    clip_start: Optional[int],
    clip_end: Optional[int],
    clips: Optional[list[Segment]],
    project_root: Optional[Path],
    clip_ts: Optional[str],
) -> Path:
    if clip_dir is not None:
        return clip_dir.expanduser().resolve()
    root = project_root or Path(".")
    ts = clip_ts or "manual"
    if clips:
        s0, s1 = clips[0], clips[-1]
        return (
            root
            / CLIP_OUTPUT_ROOT
            / f"{ts}_multi_{s0.start_capture_seq}_{s1.end_capture_seq}"
        )
    if clip_start is not None and clip_end is not None:
        return root / CLIP_OUTPUT_ROOT / f"{ts}_{clip_start}_{clip_end}"
    return root / CLIP_OUTPUT_ROOT / ts


def create_l3_params(
    *,
    fps: int = 60,
    width: int = 3840,
    height: int = 1080,
    raw_ring_seconds: Optional[float] = None,
    raw_ring_memory_mb: Optional[int] = None,
    clip_start: Optional[int] = None,
    clip_end: Optional[int] = None,
    clips_file: Optional[Path] = None,
    clip_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    clip_ts: Optional[str] = None,
) -> L3Params:
    clip_list: Optional[list[Segment]] = None
    registry: Optional[ClipRegistry] = None
    multi_session = clips_file is None and clip_start is None and clip_end is None

    if clips_file is not None:
        clip_list = load_segments_file(clips_file)
    elif clip_start is not None and clip_end is not None:
        clip_list = segment_from_single(clip_start, clip_end)

    out_dir = _resolve_clip_dir(
        clip_dir=clip_dir,
        clip_start=clip_start,
        clip_end=clip_end,
        clips=clip_list,
        project_root=project_root,
        clip_ts=clip_ts,
    )

    if clips_file is not None:
        registry = ClipRegistry.from_replay_file(clips_file, clip_dir=out_dir)
    elif multi_session:
        registry = ClipRegistry.open_session(out_dir)
    elif clip_list:
        registry = ClipRegistry.from_specs(clip_list, clip_dir=out_dir)
    else:
        raise RuntimeError("clip registry: unreachable")

    seconds = float(raw_ring_seconds) if raw_ring_seconds is not None else RAW_RING_SECONDS
    if seconds <= 0:
        raise ValueError(f"raw_ring_seconds must be > 0, got {seconds}")

    if raw_ring_memory_mb is not None:
        if int(raw_ring_memory_mb) <= 0:
            raise ValueError(f"raw_ring_memory_mb must be > 0, got {raw_ring_memory_mb}")
        memory_mb = int(raw_ring_memory_mb)
    else:
        memory_mb = raw_ring_memory_mb_for_seconds(
            seconds, width=int(width), height=int(height), fps=int(fps)
        )

    return L3Params(
        raw_ring_seconds=seconds,
        raw_ring_memory_mb=memory_mb,
        clip_start=clip_start,
        clip_end=clip_end,
        clip_dir=out_dir,
        clips=tuple(clip_list) if clip_list else None,
        clip_registry=registry,
    )


def create_l3_runtime(
    *,
    fps: int = 60,
    width: int = 3840,
    height: int = 1080,
    raw_ring_seconds: Optional[float] = None,
    raw_ring_memory_mb: Optional[int] = None,
    clip_start: Optional[int] = None,
    clip_end: Optional[int] = None,
    clips_file: Optional[Path] = None,
    clip_dir: Optional[Path] = None,
    project_root: Optional[Path] = None,
    clip_ts: Optional[str] = None,
    enable_mock_detect: bool = True,
    mock_compute_delay_sec: Optional[float] = None,
    defer_segment_detect: bool = False,
) -> "L3Runtime":
    from domain.l3.runtime import L3Runtime

    params = create_l3_params(
        fps=fps,
        width=width,
        height=height,
        raw_ring_seconds=raw_ring_seconds,
        raw_ring_memory_mb=raw_ring_memory_mb,
        clip_start=clip_start,
        clip_end=clip_end,
        clips_file=clips_file,
        clip_dir=clip_dir,
        project_root=project_root,
        clip_ts=clip_ts,
    )
    return L3Runtime(
        params,
        enable_mock_detect=enable_mock_detect,
        mock_compute_delay_sec=mock_compute_delay_sec,
        defer_segment_detect=defer_segment_detect,
    )
