"""离线滤波链：pose2d_only（ViTPose 直写）| 2D One Euro → 三角化 → 3D 滤波（legacy）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from core.geometry.calib_model import StereoCalibration
from core.geometry.kalman_pose3d import Keypoints3DKalmanFilter
from core.geometry.mmpose_tracking_2d import (
    MMPoseTracking2DConfig,
    StereoMMPoseTracking2DFilter,
)
from core.geometry.one_euro import Keypoint3DOneEuroFilter, StereoPose2DOneEuroFilter
from core.geometry.pose2sim_triangulate import (
    Pose2SimTriangulationConfig,
    TriangulationMode,
)
from core.geometry.triangulate import triangulate_matched_keypoints
from core.infer.pose3d_pipeline import one_euro_timestamp

FilterMode = Literal[
    "raw",
    "one_euro",
    "one_euro_2d",
    "one_euro_2d_kalman",
    "one_euro_2d_only",
    "mmpose_tracking_2d",
    "pose2d_only",
]


def parse_filter_mode(name: str) -> FilterMode:
    key = str(name).strip().lower().replace("+", "_")
    if key in ("pose2d_only", "2d_only", "vitpose_only"):
        return "pose2d_only"
    if key in (
        "mmpose_tracking_2d",
        "mmpose_tracking",
        "official_tracking",
        "official_tracking_2d",
        "mmpose_one_euro",
        "yolo_mmpose_tracking",
    ):
        return "mmpose_tracking_2d"
    if key in (
        "one_euro_2d_only",
        "anipose_2d",
        "phase1_one_euro",
    ):
        return "one_euro_2d_only"
    if key in ("anipose",):
        return "one_euro_2d_only"
    if key in ("raw", "none", "off"):
        return "raw"
    if key in (
        "one_euro_2d_kalman",
        "2d_kalman",
        "kalman",
        "kalman_3d",
        "one_euro_kalman",
    ):
        return "one_euro_2d_kalman"
    if key in ("one_euro_2d", "2d_one_euro", "2d", "legacy_2d"):
        return "one_euro_2d"
    if key in (
        "one_euro",
        "oneeuro",
        "full",
        "default",
        "opencap20_kalman",
        "pose2sim",
        "one_euro_pose2sim",
        "pose2sim_filter",
    ):
        return "one_euro"
    raise ValueError(
        f"未知 filter 模式: {name!r}（pose2d_only | mmpose_tracking_2d | "
        f"one_euro_2d_only | raw | one_euro_2d | one_euro_2d_kalman | one_euro）"
    )


def _pair_2d_ok(
    det_left: dict | None,
    det_right: dict | None,
    *,
    min_conf: float,
) -> bool:
    if det_left is None or det_right is None:
        return False
    kl = det_left.get("keypoints")
    kr = det_right.get("keypoints")
    if kl is None or kr is None:
        return False
    kl = np.asarray(kl, dtype=np.float64)
    kr = np.asarray(kr, dtype=np.float64)
    if kl.ndim != 2 or kr.ndim != 2 or kl.shape[0] != kr.shape[0]:
        return False
    thr = float(min_conf)
    valid = (kl[:, 2] >= thr) & (kr[:, 2] >= thr)
    return bool(np.any(valid))


@dataclass(frozen=True)
class OfflineFilterConfig:
    mode: FilterMode = "one_euro_2d"
    tri_mode: TriangulationMode = "pose2sim"
    tri_min_conf: float = 0.25
    min_conf: float = 0.25
    fps: float = 60.0
    # 2D One Euro（三角化前，像素域）
    filter_2d_min_cutoff: float = 1.0
    filter_2d_beta: float = 0.007
    filter_2d_teleport_px: float = 500.0
    tracking_thr: float = 0.3
    tracking_min_keypoints: int = 3
    # 3D One Euro（三角化后，米）
    filter_min_cutoff: float = 4.0
    filter_beta: float = 1.5
    filter_d_cutoff: float = 1.0
    filter_max_miss: int = 6
    filter_teleport_m: float = 0.5
    kalman_process_std: float = 0.002
    kalman_measure_std: float = 0.045
    kalman_output_ema: float = 0.32
    kalman_max_miss: int = 15
    kalman_measure_z_ref: float = 2.5
    kalman_measure_z_alpha: float = 0.5
    stereo_rectified: bool = False
    pose2sim_tri: Pose2SimTriangulationConfig | None = None
    calib_toml: Path | None = None


@dataclass
class OfflineFilterResult:
    kpts3d: np.ndarray | None
    tri_ok: bool
    det_left: dict | None
    det_right: dict | None
    opencap20_kpts3d: np.ndarray | None = None
    filter_2d: str | None = None


class OfflinePose3DFilterChain:
    """逐帧：2D det → 可选 One Euro(2D) → 三角化 → 可选 One Euro(3D)。"""

    def __init__(self, calib: StereoCalibration, cfg: OfflineFilterConfig) -> None:
        self._calib = calib
        self._cfg = cfg
        self._one_euro_2d: StereoPose2DOneEuroFilter | None = None
        self._mmpose_tracking_2d: StereoMMPoseTracking2DFilter | None = None
        self._one_euro_3d: Keypoint3DOneEuroFilter | None = None
        self._kalman_3d: Keypoints3DKalmanFilter | None = None
        if cfg.mode == "mmpose_tracking_2d":
            self._mmpose_tracking_2d = StereoMMPoseTracking2DFilter(
                MMPoseTracking2DConfig(
                    tracking_thr=cfg.tracking_thr,
                    min_keypoints=cfg.tracking_min_keypoints,
                    min_cutoff=cfg.filter_2d_min_cutoff,
                    beta=cfg.filter_2d_beta,
                    fps=cfg.fps,
                    use_one_euro=True,
                )
            )
        elif cfg.mode == "one_euro_2d" or cfg.mode in (
            "one_euro_2d_kalman",
            "one_euro_2d_only",
        ):
            self._one_euro_2d = StereoPose2DOneEuroFilter(
                min_cutoff=cfg.filter_2d_min_cutoff,
                beta=cfg.filter_2d_beta,
                max_miss_frames=cfg.filter_max_miss,
                teleport_px=cfg.filter_2d_teleport_px,
            )
        if cfg.mode == "one_euro_2d_kalman":
            self._kalman_3d = Keypoints3DKalmanFilter(
                fps=cfg.fps,
                process_std=cfg.kalman_process_std,
                measure_std=cfg.kalman_measure_std,
                max_miss_frames=cfg.kalman_max_miss,
                output_ema=cfg.kalman_output_ema,
                measure_z_ref=cfg.kalman_measure_z_ref,
                measure_z_alpha=cfg.kalman_measure_z_alpha,
            )
        elif cfg.mode == "one_euro":
            self._one_euro_3d = Keypoint3DOneEuroFilter(
                min_cutoff=cfg.filter_min_cutoff,
                beta=cfg.filter_beta,
                d_cutoff=cfg.filter_d_cutoff,
                max_miss_frames=cfg.filter_max_miss,
                teleport_m=cfg.filter_teleport_m,
            )

    @property
    def mode(self) -> FilterMode:
        return self._cfg.mode

    def reset(self) -> None:
        if self._one_euro_2d is not None:
            self._one_euro_2d.reset()
        if self._mmpose_tracking_2d is not None:
            self._mmpose_tracking_2d.reset()
        if self._one_euro_3d is not None:
            self._one_euro_3d.reset()
        if self._kalman_3d is not None:
            self._kalman_3d.reset()

    def process_pair(
        self,
        det_left: dict | None,
        det_right: dict | None,
        *,
        capture_seq: int,
        infer_seq: int,
    ) -> tuple[np.ndarray | None, bool, dict | None, dict | None]:
        out = self.process_pair_ex(
            det_left, det_right, capture_seq=capture_seq, infer_seq=infer_seq
        )
        return out.kpts3d, out.tri_ok, out.det_left, out.det_right

    def process_pair_ex(
        self,
        det_left: dict | None,
        det_right: dict | None,
        *,
        capture_seq: int,
        infer_seq: int,
    ) -> OfflineFilterResult:
        if self._cfg.mode == "pose2d_only":
            tri_ok = _pair_2d_ok(
                det_left, det_right, min_conf=self._cfg.min_conf
            )
            _ = capture_seq, infer_seq
            return OfflineFilterResult(
                kpts3d=None,
                tri_ok=tri_ok,
                det_left=det_left,
                det_right=det_right,
                opencap20_kpts3d=None,
            )

        if self._cfg.mode == "mmpose_tracking_2d":
            det_l, det_r = det_left, det_right
            if self._mmpose_tracking_2d is not None:
                det_l, det_r = self._mmpose_tracking_2d.filter_pair(det_l, det_r)
            tri_ok = _pair_2d_ok(det_l, det_r, min_conf=self._cfg.min_conf)
            _ = capture_seq, infer_seq
            return OfflineFilterResult(
                kpts3d=None,
                tri_ok=tri_ok,
                det_left=det_l,
                det_right=det_r,
                opencap20_kpts3d=None,
                filter_2d=StereoMMPoseTracking2DFilter.FILTER_TAG,
            )

        if self._cfg.mode == "one_euro_2d_only":
            ts = one_euro_timestamp(infer_seq, int(self._cfg.fps))
            det_l, det_r = det_left, det_right
            if self._one_euro_2d is not None:
                det_l, det_r = self._one_euro_2d.filter_pair(ts, det_l, det_r)
            tri_ok = _pair_2d_ok(det_l, det_r, min_conf=self._cfg.min_conf)
            _ = capture_seq
            return OfflineFilterResult(
                kpts3d=None,
                tri_ok=tri_ok,
                det_left=det_l,
                det_right=det_r,
                opencap20_kpts3d=None,
            )

        ts = one_euro_timestamp(infer_seq, int(self._cfg.fps))
        det_l = det_left
        det_r = det_right
        if self._one_euro_2d is not None:
            det_l, det_r = self._one_euro_2d.filter_pair(ts, det_l, det_r)

        kpts3d_raw: np.ndarray | None = None
        if det_l is not None and det_r is not None:
            kl = det_l["keypoints"]
            kr = det_r["keypoints"]
            kpts3d_raw = triangulate_matched_keypoints(
                kl[:, :2],
                kr[:, :2],
                self._calib,
                min_conf=self._cfg.tri_min_conf,
                conf_left=kl[:, 2],
                conf_right=kr[:, 2],
                mode=self._cfg.tri_mode,
                pose2sim_cfg=self._cfg.pose2sim_tri,
                calib_toml=self._cfg.calib_toml,
                stereo_rectified=self._cfg.stereo_rectified,
            )

        kpts_out = None
        if kpts3d_raw is not None:
            kpts_out = np.asarray(kpts3d_raw, dtype=np.float32)
            if self._one_euro_3d is not None:
                kpts_out = self._one_euro_3d.filter(ts, kpts_out)
            if self._kalman_3d is not None:
                kpts_out = self._kalman_3d.filter(
                    kpts_out,
                    min_conf=self._cfg.min_conf,
                )

        tri_ok = kpts_out is not None and float(np.asarray(kpts_out)[:, 3].sum()) > 0

        _ = capture_seq
        return OfflineFilterResult(
            kpts3d=kpts_out,
            tri_ok=tri_ok,
            det_left=det_l,
            det_right=det_r,
            opencap20_kpts3d=None,
        )
