"""full_session_pose2d.jsonl → Pose2Sim.triangulation() → tri_raw_3d.jsonl（路径 6）。"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from core.pose2sim.frame_range import load_jsonl_rows
from core.pose2sim.import_util import run_pose2sim_triangulation
from core.pose2sim.pose2d_export import export_pose2d_json
from core.pose2sim.pose2sim_config import Pose2SimTriConfig, write_path6_config_toml
from core.pose2sim.trc_export import _frame_rate_from_meta, _ensure_calibration
from core.pose2sim.trc_to_jsonl import write_trc_jsonl


PATH6_TRIAL_NAME = "pose2sim_path6"


@dataclass
class Pose2SimTri3DStats:
    frames_in: int = 0
    frames_out: int = 0
    trial_dir: str | None = None
    trc_path: str | None = None


def _find_tri_trc(trial_dir: Path) -> Path:
    pose3d = trial_dir / "pose-3d"
    hits = sorted(
        p
        for p in pose3d.glob("*.trc")
        if "filt" not in p.name.lower()
    )
    if not hits:
        raise RuntimeError(f"Pose2Sim triangulation 未产出 TRC（已查 {pose3d}）")
    return hits[-1]


def run_session_triangulate_pose2sim(
    *,
    pose2d_path: Path,
    out_path: Path,
    session_dir: Path,
    tri_cfg: Pose2SimTriConfig | None = None,
    frame_rate: float | None = None,
    max_frames: int = 0,
    start_seq: int | None = None,
    end_seq: int | None = None,
    trial_name: str = PATH6_TRIAL_NAME,
    calib_src: Path | None = None,
) -> Pose2SimTri3DStats:
    """jsonl → Pose2Sim 2D json → triangulation() → jsonl。"""
    pose2d_path = Path(pose2d_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    session_dir = Path(session_dir).expanduser().resolve()
    tri_cfg = tri_cfg or Pose2SimTriConfig()

    if calib_src is None:
        calib_src = (
            Path(__file__).resolve().parents[2]
            / "Pose2Sim_Full_Pipeline"
            / "3_pose2sim"
            / "Calibration"
        )
    _ensure_calibration(session_dir, Path(calib_src))

    fps = float(frame_rate or _frame_rate_from_meta(session_dir))
    trial_dir = session_dir / trial_name
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    _ensure_calibration(trial_dir, session_dir / "calibration")

    rows = load_jsonl_rows(
        pose2d_path,
        start_seq=start_seq,
        end_seq=end_seq,
        max_frames=int(max_frames),
    )
    if not rows:
        raise ValueError(f"无 2D 行: {pose2d_path}")

    n_json, _seqs = export_pose2d_json(rows, trial_dir / "pose")
    if n_json <= 0:
        raise RuntimeError("导出 Pose2Sim 2D json 失败")

    config_path = write_path6_config_toml(trial_dir, frame_rate=fps, tri=tri_cfg)

    import toml

    config_dict = toml.load(config_path)
    config_dict.setdefault("project", {})["project_dir"] = "."

    prev_cwd = Path.cwd()
    try:
        os.chdir(trial_dir)
        run_pose2sim_triangulation(config_dict)
    finally:
        os.chdir(prev_cwd)

    trc_path = _find_tri_trc(trial_dir)
    n_out = write_trc_jsonl(
        template_rows=rows,
        trc_path=trc_path,
        out_path=out_path,
        pose2sim_tri=True,
        pose2sim_filter3d=False,
    )

    return Pose2SimTri3DStats(
        frames_in=len(rows),
        frames_out=n_out,
        trial_dir=str(trial_dir),
        trc_path=str(trc_path),
    )
