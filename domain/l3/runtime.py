"""L3 读取 L1 3D 数据；判段调度。"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from domain.l1.pose_history import Pose3DSample
from domain.l3.plugins.mock_detector import MOCK_COMPUTE_DELAY_SEC, MockActionDetector

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from domain.l1.runtime import L1Runtime
    from domain.l3.params import L3Params
    from domain.l3.protocol import ActionDetector


@dataclass(frozen=True)
class L3ReadSnapshot:
    latest: Optional[Pose3DSample]
    history_len: int
    recent_samples: List[Pose3DSample] = field(default_factory=list)


@dataclass(frozen=True)
class L3AnalysisResult:
    ok: bool = True
    note: str = "analysis_not_implemented"


class L3Runtime:
    def __init__(
        self,
        params: "L3Params",
        *,
        enable_mock_detect: bool = True,
        mock_compute_delay_sec: float | None = None,
        defer_segment_detect: bool = False,
    ) -> None:
        self.params = params
        self._defer_segment_detect = bool(defer_segment_detect)
        self._last_snapshot: Optional[L3ReadSnapshot] = None
        self._last_analysis: Optional[L3AnalysisResult] = None
        self._l1: Optional["L1Runtime"] = None
        self._detector: Optional["ActionDetector"] = None
        self._post_process_done = False
        if (
            not self._defer_segment_detect
            and enable_mock_detect
            and params.clip_registry is not None
            and not params.clip_specs()
        ):
            self._detector = MockActionDetector(
                self,
                compute_delay_sec=(
                    MOCK_COMPUTE_DELAY_SEC
                    if mock_compute_delay_sec is None
                    else float(mock_compute_delay_sec)
                ),
            )

    def bind_l1(self, l1: "L1Runtime") -> None:
        self._l1 = l1

    def switch_session_dir(
        self,
        session_dir: Path,
        *,
        enable_mock_detect: bool = True,
    ) -> Path:
        """同进程切到新 session 落盘目录（相机常开切录）。"""
        from domain.registry.clip_registry import ClipRegistry

        if self._detector is not None:
            self._detector.stop()
            self._detector = None
        out = Path(session_dir).expanduser().resolve()
        out.mkdir(parents=True, exist_ok=True)
        registry = ClipRegistry.open_session(out)
        self.params.clip_dir = out
        self.params.clip_registry = registry
        self.params.clips = None
        self.params.clip_start = None
        self.params.clip_end = None
        self._post_process_done = False
        self._last_snapshot = None
        self._last_analysis = None
        if (
            self._defer_segment_detect
            and enable_mock_detect
            and registry is not None
        ):
            # post_process@exit：不在 live 启 detector
            pass
        elif enable_mock_detect and registry is not None and not self.params.clip_specs():
            self._detector = MockActionDetector(
                self,
                compute_delay_sec=MOCK_COMPUTE_DELAY_SEC,
            )
        return out

    @property
    def last_snapshot(self) -> Optional[L3ReadSnapshot]:
        return self._last_snapshot

    def read_l1_pose3d(self, l1: "L1Runtime") -> L3ReadSnapshot:
        snap = L3ReadSnapshot(
            latest=l1.latest_pose3d_sample(),
            history_len=l1.pose3d_history_len(),
            recent_samples=l1.snapshot_pose3d(),
        )
        self._last_snapshot = snap
        return snap

    def analyze_l1_data(self, snapshot: L3ReadSnapshot) -> L3AnalysisResult:
        if snapshot.history_len == 0:
            return L3AnalysisResult(ok=False, note="no_l1_history")
        return L3AnalysisResult()

    def register_action_segment(
        self,
        start_capture_seq: int,
        end_capture_seq: int,
        *,
        label: Optional[str] = None,
    ) -> int:
        idx = self.params.append_clip_segment(
            start_capture_seq, end_capture_seq, label=label
        )
        if self._l1 is not None:
            self._l1.notify_segment_registered(
                start_capture_seq,
                end_capture_seq,
                label=label,
                index=idx,
            )
        return idx

    def register_action_marker(
        self,
        capture_seq: int,
        *,
        label: Optional[str] = None,
        kind: str = "marker",
    ) -> int:
        return self.params.append_clip_marker(
            capture_seq, label=label, kind=kind
        )

    def on_l1_pose3d_frame(self, l1: "L1Runtime") -> L3AnalysisResult:
        if self._defer_segment_detect:
            return L3AnalysisResult(note="segment_detect_deferred")
        snap = self.read_l1_pose3d(l1)
        result = self.analyze_l1_data(snap)
        self._last_analysis = result
        if self._detector is not None and snap.latest is not None:
            self._detector.on_capture_seq(snap.latest.capture_seq)
        return result

    def finalize_session(self, l1: "L1Runtime") -> list[tuple[int, int, Optional[str]]]:
        """session 模式：退出时读 session_pose.jsonl 判段（不读内存历史）。"""
        if self._post_process_done:
            return []
        self._post_process_done = True
        if self._detector is not None:
            self._detector.stop()
            self._detector = None

        if not self._defer_segment_detect:
            return []

        clip_dir = self.params.clip_dir
        if clip_dir is None:
            return []

        if self.params.clip_specs():
            logger.info("session post_process: skip (predefined clip specs)")
            return []

        if l1._pose_log is not None:
            l1.flush_pose_log()

        from domain.l3.session_post_process import run_mock_session_post_process

        return run_mock_session_post_process(
            self,
            pose_path=Path(clip_dir) / "session_pose.jsonl",
        )

    def stop(self) -> None:
        if self._detector is not None:
            self._detector.stop()
            self._detector = None

    def display_status_suffix(self) -> str:
        if self._defer_segment_detect:
            return "L3:post_process@exit"
        snap = self._last_snapshot
        if snap is None or snap.latest is None:
            return "L3:—"
        tri = "ok" if snap.latest.tri_ok else "—"
        return f"L3:hist={snap.history_len} cap={snap.latest.capture_seq} tri={tri}"
