"""OpenCap PreAugmentation TRC → Pose2Sim.filtering()（LSTM 之前）。"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from core.pose2sim.import_util import ensure_pose2sim_import

Pose2SimFilterType = Literal[
    "butterworth",
    "kalman",
    "one_euro",
    "gcv_spline",
    "butterworth_on_speed",
    "gaussian",
    "loess",
    "median",
]


@dataclass(frozen=True)
class Pose2SimFilterConfig:
    reject_outliers: bool = True
    filter: bool = True
    filter_type: Pose2SimFilterType = "butterworth"
    butterworth_order: int = 4
    butterworth_cutoff_hz: float = 6.0
    kalman_trust_ratio: int = 500
    kalman_smooth: bool = True
    one_euro_cutoff_hz: float = 4.0
    one_euro_beta: float = 1.5
    one_euro_d_cutoff_hz: float = 1.0


def _write_filter_config(
    work_dir: Path,
    *,
    frame_rate: float,
    cfg: Pose2SimFilterConfig,
) -> Path:
    config_path = work_dir / "Config.toml"
    text = f"""# OpenCap PreAugmentation → Pose2Sim 3D filtering（LSTM 之前）

[project]
project_dir = '.'
multi_person = false
frame_rate = {float(frame_rate):g}
frame_range = 'all'

[filtering]
reject_outliers = {'true' if cfg.reject_outliers else 'false'}
filter = {'true' if cfg.filter else 'false'}
type = '{cfg.filter_type}'
display_figures = false
save_filt_plots = false
make_c3d = false

[filtering.butterworth]
order = {int(cfg.butterworth_order)}
cut_off_frequency = {float(cfg.butterworth_cutoff_hz):g}

[filtering.kalman]
trust_ratio = {int(cfg.kalman_trust_ratio)}
smooth = {'true' if cfg.kalman_smooth else 'false'}

[filtering.one_euro]
cut_off_frequency = {float(cfg.one_euro_cutoff_hz):g}
beta = {float(cfg.one_euro_beta):g}
d_cut_off_frequency = {float(cfg.one_euro_d_cutoff_hz):g}

[logging]
use_custom_logging = true
"""
    config_path.write_text(text, encoding="utf-8")
    return config_path


def filter_preaug_trc(
    input_trc: Path,
    *,
    frame_rate: float,
    cfg: Pose2SimFilterConfig | None = None,
    work_dir: Path | None = None,
) -> Path:
    """
    对 PreAugmentation TRC 跑 Pose2Sim.filtering()，返回滤后 TRC 路径。

    滤后文件复制到 ``input_trc.parent/PreAugmentation_filt_<type>.trc``。
    """
    input_trc = Path(input_trc).expanduser().resolve()
    if not input_trc.is_file():
        raise FileNotFoundError(f"找不到 TRC: {input_trc}")

    cfg = cfg or Pose2SimFilterConfig()
    work_dir = Path(work_dir or input_trc.parent / "pose2sim_3d").expanduser().resolve()
    pose3d_dir = work_dir / "pose-3d"
    if work_dir.exists():
        shutil.rmtree(work_dir)
    pose3d_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_trc, pose3d_dir / input_trc.name)

    config_path = _write_filter_config(work_dir, frame_rate=frame_rate, cfg=cfg)

    import toml
    from core.pose2sim.import_util import run_pose2sim_filtering

    config_dict = toml.load(config_path)
    config_dict.setdefault("project", {})["project_dir"] = "."

    prev_cwd = Path.cwd()
    try:
        os.chdir(work_dir)
        run_pose2sim_filtering(config_dict)
    finally:
        os.chdir(prev_cwd)

    # Pose2Sim 用全路径 split('_') 写输出，可能落在 opencap/ 而非 pose-3d/
    search_dirs = (pose3d_dir, work_dir, input_trc.parent, work_dir.parent)
    filt_files: list[Path] = []
    for root in search_dirs:
        filt_files.extend(sorted(Path(root).glob("*_filt_*.trc")))
    filt_files = sorted(set(filt_files))
    if not filt_files:
        raise RuntimeError(
            f"Pose2Sim filtering 未产出 *_filt_*.trc（已查 {search_dirs}）"
        )

    out_path = input_trc.parent / f"PreAugmentation_filt_{cfg.filter_type}.trc"
    shutil.copy2(filt_files[-1], out_path)
    return out_path


def resolve_lstm_input_trc(opencap_dir: Path) -> Path:
    """LSTM 输入：优先滤后 TRC，否则回退原始 PreAugmentation.trc。"""
    opencap_dir = Path(opencap_dir).expanduser().resolve()
    patterns = (
        "PreAugmentation_filt_*.trc",
        "PreAugmentation_*_filt_*.trc",
        "*_filt_*.trc",
    )
    for pattern in patterns:
        hits = sorted(opencap_dir.glob(pattern))
        if hits:
            return hits[-1]
    work_pose3d = opencap_dir / "pose2sim_3d" / "pose-3d"
    if work_pose3d.is_dir():
        hits = sorted(work_pose3d.glob("*_filt_*.trc"))
        if hits:
            return hits[-1]
    raw = opencap_dir / "PreAugmentation.trc"
    if raw.is_file():
        return raw
    raise FileNotFoundError(f"opencap/ 下找不到 PreAugmentation TRC: {opencap_dir}")
