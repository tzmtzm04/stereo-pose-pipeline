"""双目 2D infer + 三角化 → Pose3D。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from core.geometry.calib_model import StereoCalibration
from core.geometry.one_euro import Keypoint3DOneEuroFilter, StereoPose2DOneEuroFilter
from core.geometry.pose2sim_triangulate import TriangulationMode
from core.geometry.triangulate import triangulate_matched_keypoints
from core.infer.stereo_pipeline import EyePoseResult, StereoInferPipeline, StereoInferResult


def _best_person(eye: EyePoseResult) -> Optional[dict]:
    if not eye.poses:
        return None
    return max(eye.poses, key=lambda p: float(p.get("conf", 0.0)))


def _copy_det(det: dict | None) -> dict | None:
    if det is None:
        return None
    out = dict(det)
    kp = det.get("keypoints")
    if kp is not None:
        out["keypoints"] = np.asarray(kp, dtype=np.float32).copy()
    return out


def infer_timestamp(infer_seq: int, fps: int) -> float:
    return float(infer_seq) / max(float(fps), 1.0)


def capture_timestamp(capture_seq: int, fps: int) -> float:
    """One Euro 时间轴：与采集 capture_seq 对齐（跳帧时勿用 infer frame_id）。"""
    return float(capture_seq) / max(float(fps), 1.0)


@dataclass
class StereoPose3DResult:
    infer: StereoInferResult
    kpts3d: Optional[np.ndarray]  # (17, 4)
    tri_ok: bool
    det_left: Optional[dict] = None
    det_right: Optional[dict] = None
    tri_ms: float = 0.0


class StereoPose3DPipeline:
    """StereoInferPipeline + Pose2Sim 三角化 + 可选 One Euro(2D/3D)。"""

    def __init__(
        self,
        infer_pipeline: StereoInferPipeline,
        calib: StereoCalibration,
        *,
        min_conf: float = 0.0,
        tri_mode: TriangulationMode = "opencv",
        use_3d_filter: bool = False,
        use_2d_filter: bool = True,
        filter_2d_min_cutoff: float = 1.0,
        filter_2d_beta: float = 0.007,
        filter_min_cutoff: float = 4.0,
        filter_beta: float = 1.5,
        filter_d_cutoff: float = 1.0,
        filter_max_miss: int = 6,
        filter_teleport_m: float = 0.5,
        fps: int = 60,
    ) -> None:
        self._infer = infer_pipeline
        self._calib = calib
        self._min_conf = float(min_conf)
        self._tri_mode: TriangulationMode = (
            tri_mode if tri_mode in ("opencv", "pose2sim") else "opencv"
        )
        self._fps = max(1, int(fps))
        self._filter_2d: StereoPose2DOneEuroFilter | None = None
        self._filter_3d: Keypoint3DOneEuroFilter | None = None
        if use_2d_filter:
            self._filter_2d = StereoPose2DOneEuroFilter(
                min_cutoff=float(filter_2d_min_cutoff),
                beta=float(filter_2d_beta),
                max_miss_frames=int(filter_max_miss),
            )
        if use_3d_filter:
            self._filter_3d = Keypoint3DOneEuroFilter(
                min_cutoff=float(filter_min_cutoff),
                beta=float(filter_beta),
                d_cutoff=float(filter_d_cutoff),
                max_miss_frames=int(filter_max_miss),
                teleport_m=float(filter_teleport_m),
            )

    @property
    def infer_pipeline(self) -> StereoInferPipeline:
        return self._infer

    def process_sbs(
        self,
        sbs: Any,
        capture_ms: float,
        *,
        timestamp: float = 0.0,
        parallel_eyes: bool = True,
        release_sbs: bool = True,
        with_gpu_preview: bool = False,
        pipeline_overlap: bool = False,
        prefetch_sbs: Any | None = None,
    ) -> Optional[StereoPose3DResult]:
        infer_res = self._infer.process_sbs(
            sbs,
            capture_ms,
            parallel_eyes=parallel_eyes,
            release_sbs=release_sbs,
            with_gpu_preview=with_gpu_preview,
            pipeline_overlap=pipeline_overlap,
            prefetch_sbs=prefetch_sbs,
        )
        if infer_res is None:
            return None
        return self._triangulate(infer_res, timestamp=timestamp)

    def process_vic_bgr_prep(
        self,
        prep_item: Any,
        *,
        timestamp: float = 0.0,
        parallel_eyes: bool = True,
        prep_q_skipped: int = 0,
    ) -> Optional[StereoPose3DResult]:
        from core.infer.vic_bgr_stereo import VicBgrInferPrepItem

        if not isinstance(prep_item, VicBgrInferPrepItem):
            raise TypeError(f"期望 VicBgrInferPrepItem，收到 {type(prep_item)!r}")
        infer_res = self._infer.process_vic_bgr_prep(
            prep_item.prep,
            prep_item.capture_ms,
            parallel_eyes=parallel_eyes,
            prep_q_skipped=prep_q_skipped,
        )
        return self._triangulate(infer_res, timestamp=timestamp)

    def process_prepared(
        self,
        prep: Any,
        *,
        timestamp: float = 0.0,
        parallel_eyes: bool = True,
        release: bool = True,
    ) -> Optional[StereoPose3DResult]:
        infer_res = self._infer.process_prepared(
            prep,
            parallel_eyes=parallel_eyes,
            release=release,
        )
        if infer_res is None:
            return None
        return self._triangulate(infer_res, timestamp=timestamp)

    def process_capture(
        self,
        cap: Any,
        *,
        timestamp: float = 0.0,
        parallel_eyes: bool = True,
    ) -> Optional[StereoPose3DResult]:
        infer_res = self._infer.process_capture(cap, parallel_eyes=parallel_eyes)
        if infer_res is None:
            return None
        return self._triangulate(infer_res, timestamp=timestamp)

    def _triangulate(
        self,
        infer_res: StereoInferResult,
        *,
        timestamp: float,
    ) -> StereoPose3DResult:
        t_tri = time.perf_counter()
        cap_seq = int(getattr(infer_res, "capture_seq", -1))
        if cap_seq < 0:
            cap_seq = int(infer_res.frame_id)
        ts = (
            timestamp
            if timestamp > 0.0
            else capture_timestamp(cap_seq, self._fps)
        )
        det_l_raw = _best_person(infer_res.left)
        det_r_raw = _best_person(infer_res.right)
        det_l_tri = det_l_raw
        det_r_tri = det_r_raw
        if self._filter_2d is not None:
            det_l_tri, det_r_tri = self._filter_2d.filter_pair(
                ts,
                _copy_det(det_l_raw),
                _copy_det(det_r_raw),
            )

        kpts3d: Optional[np.ndarray] = None
        tri_ok = False
        if det_l_tri is not None and det_r_tri is not None:
            kl = det_l_tri["keypoints"][:, :2]
            kr = det_r_tri["keypoints"][:, :2]
            cl = det_l_tri["keypoints"][:, 2]
            cr = det_r_tri["keypoints"][:, 2]
            kpts3d = triangulate_matched_keypoints(
                kl,
                kr,
                self._calib,
                min_conf=self._min_conf,
                conf_left=cl,
                conf_right=cr,
                mode=self._tri_mode,
            )
            if self._filter_3d is not None and kpts3d is not None:
                kpts3d = self._filter_3d.filter(ts, kpts3d)
            tri_ok = kpts3d is not None and float(kpts3d[:, 3].sum()) > 0
        tri_ms = (time.perf_counter() - t_tri) * 1000.0

        return StereoPose3DResult(
            infer=infer_res,
            kpts3d=kpts3d,
            tri_ok=tri_ok,
            det_left=det_l_tri,
            det_right=det_r_tri,
            tri_ms=tri_ms,
        )

    def close(self) -> None:
        self._infer.close()


one_euro_timestamp = infer_timestamp
