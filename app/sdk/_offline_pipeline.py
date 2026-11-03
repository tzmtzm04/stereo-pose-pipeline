"""离线黑盒 SDK 入口：run_offline_pipeline_from_session.py（Stage 0–3，DLT）。"""

from __future__ import annotations

from pathlib import Path

from app.paths import ROOT


def build_offline_pipeline_cmd(
    session_dir: Path,
    *,
    output_dir: Path | None = None,
    skip_infer: bool = False,
    tri_backend: str = "dlt",
    max_frames: int = 0,
    log_every: int = 30,
) -> list[str]:
    """整链离线黑盒（合作方 verify 同路径）。

    session_dir：SessionBundle 根（route2_clips 或 motion_segments 伪 bundle）
    output_dir：产出目录；默认与 session_dir 相同（整 session 在 outputs/offline 时）
    """
    session_dir = session_dir.expanduser().resolve()
    cmd = [
        "python3",
        str(ROOT / "scripts/offline/run_offline_pipeline_from_session.py"),
        "--session",
        str(session_dir),
        "--tri-backend",
        str(tri_backend),
        "--log-every",
        str(int(log_every)),
    ]
    if output_dir is not None:
        out = output_dir.expanduser().resolve()
        cmd.extend(["--output-dir", str(out)])
    if skip_infer:
        cmd.append("--skip-infer")
    if max_frames > 0:
        cmd.extend(["--max-frames", str(int(max_frames))])
    return cmd
