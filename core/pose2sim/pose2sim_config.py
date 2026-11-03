"""Pose2Sim Config.toml 片段（路径 6：triangulation + filtering）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.opencap.trc_pose2sim_filter import Pose2SimFilterConfig


@dataclass(frozen=True)
class Pose2SimTriConfig:
    reproj_error_threshold_px: float = 15.0
    likelihood_threshold: float = 0.3
    min_cameras_for_triangulation: int = 2
    interp_if_gap_smaller_than: int = 20
    interpolation: str = "linear"
    min_chunk_size: int = 10
    sections_to_keep: str = "all"
    fill_large_gaps_with: str = "last_value"
    remove_incomplete_frames: bool = False
    undistort_points: bool = False
    make_c3d: bool = False
    show_interp_indices: bool = False


def write_path6_config_toml(
    trial_dir: Path,
    *,
    frame_rate: float,
    tri: Pose2SimTriConfig | None = None,
    filt: Pose2SimFilterConfig | None = None,
) -> Path:
    tri = tri or Pose2SimTriConfig()
    filt = filt or Pose2SimFilterConfig()
    text = f"""# 路径 6 · Pose2Sim triangulation + filtering（原生管线）

[project]
project_dir = '.'
multi_person = false
frame_rate = {float(frame_rate):g}
frame_range = 'all'

[pose]
pose_model = 'Body'
overwrite_pose = false

[calibration]
calibration_type = 'convert'

[personAssociation.single_person]
tracked_keypoint = 'Nose'

[triangulation]
reproj_error_threshold_triangulation = {float(tri.reproj_error_threshold_px):g}
likelihood_threshold_triangulation = {float(tri.likelihood_threshold):g}
min_cameras_for_triangulation = {int(tri.min_cameras_for_triangulation)}
interp_if_gap_smaller_than = {int(tri.interp_if_gap_smaller_than)}
interpolation = '{tri.interpolation}'
min_chunk_size = {int(tri.min_chunk_size)}
sections_to_keep = '{tri.sections_to_keep}'
fill_large_gaps_with = '{tri.fill_large_gaps_with}'
remove_incomplete_frames = {'true' if tri.remove_incomplete_frames else 'false'}
undistort_points = {'true' if tri.undistort_points else 'false'}
make_c3d = {'true' if tri.make_c3d else 'false'}
show_interp_indices = {'true' if tri.show_interp_indices else 'false'}

[filtering]
reject_outliers = {'true' if filt.reject_outliers else 'false'}
filter = {'true' if filt.filter else 'false'}
type = '{filt.filter_type}'
display_figures = false
save_filt_plots = false
make_c3d = false

[filtering.butterworth]
order = {int(filt.butterworth_order)}
cut_off_frequency = {float(filt.butterworth_cutoff_hz):g}

[filtering.kalman]
trust_ratio = {int(filt.kalman_trust_ratio)}
smooth = {'true' if filt.kalman_smooth else 'false'}

[filtering.one_euro]
cut_off_frequency = {float(filt.one_euro_cutoff_hz):g}
beta = {float(filt.one_euro_beta):g}
d_cut_off_frequency = {float(filt.one_euro_d_cutoff_hz):g}

[logging]
use_custom_logging = true
"""
    path = Path(trial_dir) / "Config.toml"
    path.write_text(text, encoding="utf-8")
    return path
