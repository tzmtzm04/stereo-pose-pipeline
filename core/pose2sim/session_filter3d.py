"""tri_raw_3d.jsonl / pose2sim_path6 TRC → Pose2Sim.filtering() → tri_filtered_3d.jsonl（路径 6）。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import toml

from core.opencap.trc_pose2sim_filter import Pose2SimFilterConfig, filter_preaug_trc
from core.pose2sim.frame_range import load_jsonl_rows
from core.pose2sim.import_util import run_pose2sim_filtering
from core.pose2sim.session_triangulate import PATH6_TRIAL_NAME
from core.pose2sim.trc_export import _frame_rate_from_meta, export_rows_to_trc
from core.pose2sim.trc_to_jsonl import TrcYupLayout, write_trc_jsonl


@dataclass
class Pose2SimFilter3DStats:
    frames_in_raw: int = 0
    frames_in: int = 0
    frames_out: int = 0
    trc_in: str | None = None
    trc_filt: str | None = None


def _find_raw_trc(pose3d_dir: Path) -> Path | None:
    hits = sorted(
        p for p in pose3d_dir.glob("*.trc") if "filt" not in p.name.lower()
    )
    return hits[-1] if hits else None


def filter_trc_in_trial(
    trial_dir: Path,
    *,
    frame_rate: float,
    cfg: Pose2SimFilterConfig | None = None,
) -> Path:
    """对已三角化的 trial（pose-3d/*.trc）跑 Pose2Sim.filtering()。"""
    trial_dir = Path(trial_dir).expanduser().resolve()
    cfg = cfg or Pose2SimFilterConfig()
    pose3d_dir = trial_dir / "pose-3d"
    trc_in = _find_raw_trc(pose3d_dir)
    if trc_in is None:
        raise FileNotFoundError(f"找不到 raw TRC: {pose3d_dir}")

    config_path = trial_dir / "Config.toml"
    if config_path.is_file():
        config_dict = toml.load(config_path)
    else:
        config_dict = {}
    config_dict.setdefault("project", {})["project_dir"] = "."
    config_dict.setdefault("project", {})["frame_rate"] = float(frame_rate)
    filt = config_dict.setdefault("filtering", {})
    filt["reject_outliers"] = bool(cfg.reject_outliers)
    filt["filter"] = bool(cfg.filter)
    filt["type"] = cfg.filter_type
    filt["display_figures"] = False
    filt["save_filt_plots"] = False
    filt["make_c3d"] = False
    config_dict.setdefault("filtering", {}).setdefault("butterworth", {})
    config_dict["filtering"]["butterworth"]["order"] = int(cfg.butterworth_order)
    config_dict["filtering"]["butterworth"]["cut_off_frequency"] = float(
        cfg.butterworth_cutoff_hz
    )
    config_path.write_text(toml.dumps(config_dict), encoding="utf-8")

    prev_cwd = Path.cwd()
    try:
        os.chdir(trial_dir)
        run_pose2sim_filtering(config_dict)
    finally:
        os.chdir(prev_cwd)

    filt_files = sorted(pose3d_dir.glob("*_filt_*.trc"))
    if not filt_files:
        raise RuntimeError(
            f"Pose2Sim filtering 未产出 *_filt_*.trc（已查 {pose3d_dir}）"
        )
    return filt_files[-1]


def run_session_filter3d_pose2sim(
    *,
    pose3d_path: Path,
    out_path: Path,
    session_dir: Path,
    cfg: Pose2SimFilterConfig | None = None,
    frame_rate: float | None = None,
    max_frames: int = 0,
    start_seq: int | None = None,
    end_seq: int | None = None,
    trial_dir: Path | None = None,
) -> Pose2SimFilter3DStats:
    """Pose2Sim Hampel + Butterworth → jsonl（优先用 path6 trial 内 TRC）。"""
    pose3d_path = Path(pose3d_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    session_dir = Path(session_dir).expanduser().resolve()
    cfg = cfg or Pose2SimFilterConfig()

    raw_count = sum(1 for _ in pose3d_path.open(encoding="utf-8"))
    rows = load_jsonl_rows(
        pose3d_path,
        start_seq=start_seq,
        end_seq=end_seq,
        max_frames=int(max_frames),
        require_kpts3d=True,
    )
    if not rows:
        raise ValueError(f"无有效 3D 行: {pose3d_path}")

    fps = float(frame_rate or _frame_rate_from_meta(session_dir))
    trial = Path(trial_dir or session_dir / PATH6_TRIAL_NAME).expanduser().resolve()
    trc_in_path: Path | None = _find_raw_trc(trial / "pose-3d") if trial.is_dir() else None

    trc_yup_layout: TrcYupLayout = "scheme_b"
    if trc_in_path is not None and trial.is_dir():
        filt_trc = filter_trc_in_trial(trial, frame_rate=fps, cfg=cfg)
        trc_in = str(trc_in_path)
        trc_yup_layout = "pose2sim_native"
    else:
        work = session_dir / "pose2sim_3d"
        work.mkdir(parents=True, exist_ok=True)
        trc_in_local = work / "tri_raw.trc"
        n_trc, _seqs = export_rows_to_trc(rows, trc_in_local, frame_rate=fps)
        if n_trc <= 0:
            raise RuntimeError("导出 TRC 失败：无有效 3D marker")
        filt_trc = filter_preaug_trc(
            trc_in_local,
            frame_rate=fps,
            cfg=cfg,
            work_dir=work / "p2s_work",
        )
        trc_in = str(trc_in_local)

    n_out = write_trc_jsonl(
        template_rows=rows,
        trc_path=filt_trc,
        out_path=out_path,
        pose2sim_tri=True,
        pose2sim_filter3d=True,
        trc_yup_layout=trc_yup_layout,
    )

    return Pose2SimFilter3DStats(
        frames_in_raw=raw_count,
        frames_in=len(rows),
        frames_out=n_out,
        trc_in=trc_in,
        trc_filt=str(filt_trc),
    )
