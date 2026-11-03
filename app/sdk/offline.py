"""离线黑盒 SDK：整 session 或按运动段提交 offline_pipeline Job。"""

from __future__ import annotations

import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path

from app.paths import CLIP_OUTPUT_ROOT, OUTPUTS, ROOT
from app.sdk._finalize import load_motion_segment_refs
from app.sdk._offline_pipeline import build_offline_pipeline_cmd
from app.sdk.types import JobStatus, MotionSegmentRef, OfflineArtifacts
from contracts.segment import Segment


@dataclass
class JobHandle:
    session_id: str
    offline_dir: Path
    _proc: subprocess.Popen[str] | None = None
    _thread: threading.Thread | None = None
    _error: str | None = None
    _done: bool = False

    def poll(self) -> JobStatus:
        if self._error is not None:
            return JobStatus(state="failed", error=self._error)
        if self._done:
            return JobStatus(state="done")
        if self._proc is not None:
            code = self._proc.poll()
            if code is None:
                return JobStatus(state="running", stage=2)
            if code != 0:
                return JobStatus(state="failed", error=f"exit {code}")
            self._done = True
            return JobStatus(state="done")
        if self._thread is not None and self._thread.is_alive():
            return JobStatus(state="running", stage=0)
        if self._error is not None:
            return JobStatus(state="failed", error=self._error)
        if self._done:
            return JobStatus(state="done")
        return JobStatus(state="pending")

    def wait(self, timeout: float | None = None) -> JobStatus:
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self._proc is not None:
            try:
                self._proc.wait(timeout=timeout)
            except Exception:
                pass
        return self.poll()

    def artifacts(self) -> OfflineArtifacts:
        d = self.offline_dir
        summaries: dict[str, Path] = {}
        for name in (
            "offline_pipeline_summary.json",
            "clean_2d_summary.json",
            "tri_filtered_3d_summary.json",
            "tri_raw_3d_summary.json",
        ):
            p = d / name
            if p.is_file():
                summaries[name] = p
        for p in sorted(d.glob("offline_pipeline*_summary.json")):
            summaries.setdefault(p.name, p)

        def _maybe(path: Path) -> Path | None:
            return path if path.is_file() else None

        tri = _maybe(d / "tri_filtered_3d.jsonl")
        return OfflineArtifacts(
            session_dir=d,
            pose2d=_maybe(d / "full_session_pose2d.jsonl"),
            clean_2d=_maybe(d / "clean_2d.jsonl"),
            tri_filtered_3d=tri,
            constrained_3d=tri,
            summaries=summaries,
        )


@dataclass
class OfflineJob:
    offline_root: Path = field(default_factory=lambda: OUTPUTS / "offline")

    def submit(
        self,
        session_id: str,
        *,
        segments: list[Segment] | None = None,
        with_ik: bool = False,
    ) -> JobHandle:
        """整 session 离线黑盒（Stage0–3，DLT）。"""
        _ = segments
        if with_ik:
            print(
                "sdk OfflineJob: with_ik 已忽略（离线黑盒 Stage0–3 不含 IK）",
                flush=True,
            )
        sid = str(session_id).strip()
        live_dir = CLIP_OUTPUT_ROOT / sid
        if not live_dir.is_dir():
            raise FileNotFoundError(f"找不到实时 session: {live_dir}")

        offline_dir = Path(self.offline_root).expanduser().resolve() / sid
        offline_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_offline_pipeline_cmd(live_dir, output_dir=offline_dir)
        print(f"sdk OfflineJob submit: {' '.join(cmd)}", flush=True)
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return JobHandle(session_id=sid, offline_dir=offline_dir, _proc=proc)

    def submit_motion_segments(
        self,
        session_id: str,
        *,
        with_ik: bool = False,
    ) -> list[JobHandle]:
        """对 finalize 切出的每个运动段各跑一条离线黑盒链。

        输入：route2_clips/{id}/motion_segments/{stem}/（伪 SessionBundle）
        产出：outputs/offline/{stem}/
        """
        sid = str(session_id).strip()
        live_dir = CLIP_OUTPUT_ROOT / sid
        refs = load_motion_segment_refs(live_dir)
        if not refs:
            raise FileNotFoundError(
                f"无 motion_segments.json：先 RealtimeSession.finalize() → {live_dir}"
            )
        return [self.submit_clip(ref, with_ik=with_ik) for ref in refs]

    def submit_clip(
        self,
        ref: MotionSegmentRef,
        *,
        with_ik: bool = False,
    ) -> JobHandle:
        """单条运动段：run_offline_pipeline（--session 绝对路径 + --output-dir）。"""
        if with_ik:
            print(
                "sdk OfflineJob clip: with_ik 已忽略（离线黑盒 Stage0–3 不含 IK）",
                flush=True,
            )
        clip_dir = Path(ref.clip_dir).expanduser().resolve()
        if not clip_dir.is_dir():
            raise FileNotFoundError(f"运动段目录不存在: {clip_dir}")

        offline_dir = Path(self.offline_root).expanduser().resolve() / clip_dir.name
        offline_dir.mkdir(parents=True, exist_ok=True)

        holder: dict[str, str | None] = {"error": None}

        def _run() -> None:
            try:
                cmd = build_offline_pipeline_cmd(
                    clip_dir,
                    output_dir=offline_dir,
                )
                print(f"sdk OfflineJob clip: {' '.join(cmd)}", flush=True)
                proc = subprocess.run(
                    cmd,
                    cwd=str(ROOT),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if proc.returncode != 0:
                    holder["error"] = (
                        proc.stderr.strip()
                        or proc.stdout.strip()
                        or f"exit {proc.returncode}"
                    )[-800:]
            except Exception as exc:
                holder["error"] = str(exc)

        thread = threading.Thread(
            target=_run, name=f"sdk_offline_clip_{ref.stem}", daemon=True
        )
        handle = JobHandle(
            session_id=ref.stem,
            offline_dir=offline_dir,
            _thread=thread,
        )

        def _watch() -> None:
            thread.join()
            handle._error = holder["error"]
            handle._done = holder["error"] is None

        thread.start()
        threading.Thread(target=_watch, daemon=True).start()
        return handle
