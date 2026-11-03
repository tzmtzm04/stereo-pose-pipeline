"""双目 2D 关键点 → 3D（左相机系，米）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from core.geometry.calib_model import StereoCalibration
from core.offline.two_stage.coco_wholebody_layout import skip_wholebody_hand_triangulation
from core.geometry.pose2sim_triangulate import (
    Pose2SimTriangulationConfig,
    TriangulationMode,
    parse_triangulation_mode,
    triangulate_matched_keypoints_pose2sim,
)
from core.geometry.stereo_rectify import (
    projection_matrices_rectified,
    rectified_to_left_cam_opencv,
)


def triangulate_matched_keypoints(
    kpts_left: np.ndarray,
    kpts_right: np.ndarray,
    calib: StereoCalibration,
    *,
    min_conf: float = 0.0,
    conf_left: np.ndarray | None = None,
    conf_right: np.ndarray | None = None,
    mode: TriangulationMode = "pose2sim",
    pose2sim_cfg: Pose2SimTriangulationConfig | None = None,
    calib_toml: Path | str | None = None,
    stereo_rectified: bool = False,
    exclude_wholebody_hands: bool = True,
) -> np.ndarray:
    tri_mode = parse_triangulation_mode(mode)
    if tri_mode == "pose2sim":
        return triangulate_matched_keypoints_pose2sim(
            kpts_left,
            kpts_right,
            calib,
            min_conf=min_conf,
            conf_left=conf_left,
            conf_right=conf_right,
            cfg=pose2sim_cfg,
            calib_toml=calib_toml,
            stereo_rectified=stereo_rectified,
            exclude_wholebody_hands=exclude_wholebody_hands,
        )
    kl = np.asarray(kpts_left, dtype=np.float64)
    kr = np.asarray(kpts_right, dtype=np.float64)
    if kl.shape != kr.shape:
        raise ValueError(f"左右关键点形状不一致: {kl.shape} vs {kr.shape}")

    single = kl.ndim == 2
    if single:
        kl = kl[np.newaxis, ...]
        kr = kr[np.newaxis, ...]
        if conf_left is not None:
            conf_left = np.asarray(conf_left, dtype=np.float32).reshape(1, -1)
        if conf_right is not None:
            conf_right = np.asarray(conf_right, dtype=np.float32).reshape(1, -1)

    p_count, j_count, _ = kl.shape
    out = np.zeros((p_count, j_count, 4), dtype=np.float32)

    K1, D1 = calib.K1, calib.D1
    K2, D2 = calib.K2, calib.D2
    R, T = calib.R, calib.T
    if stereo_rectified:
        p1, p2 = projection_matrices_rectified(calib)
        r1 = calib.R1
    else:
        p1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
        p2 = K2 @ np.hstack([R, T])
        r1 = None

    for pi in range(p_count):
        for ji in range(j_count):
            if skip_wholebody_hand_triangulation(
                ji, j_count, enabled=exclude_wholebody_hands
            ):
                continue
            c_l = float(conf_left[pi, ji]) if conf_left is not None else 1.0
            c_r = float(conf_right[pi, ji]) if conf_right is not None else 1.0
            conf = min(c_l, c_r)
            if conf < min_conf:
                continue
            pl = kl[pi, ji]
            pr = kr[pi, ji]
            if pl[0] <= 0 or pr[0] <= 0:
                continue
            if stereo_rectified:
                pl_pt = pl[:2].reshape(2, 1)
                pr_pt = pr[:2].reshape(2, 1)
            else:
                pl_pt = cv2.undistortPoints(
                    pl.reshape(1, 1, 2), K1, D1, None, K1
                ).reshape(2, 1)
                pr_pt = cv2.undistortPoints(
                    pr.reshape(1, 1, 2), K2, D2, None, K2
                ).reshape(2, 1)
            hom = cv2.triangulatePoints(p1, p2, pl_pt, pr_pt)
            if abs(hom[3, 0]) < 1e-9:
                continue
            xyz = hom[:3, 0] / hom[3, 0]
            if stereo_rectified and r1 is not None:
                xyz = rectified_to_left_cam_opencv(xyz, r1)
            # P1/P2 来自 stereoRectify（T 以 mm 计）→ 与 opencv 原路径（米）对齐
            if stereo_rectified:
                xyz = xyz / 1000.0
            xyz = xyz.astype(np.float32)
            out[pi, ji, 0] = xyz[0]
            out[pi, ji, 1] = xyz[1]
            out[pi, ji, 2] = xyz[2]
            out[pi, ji, 3] = conf

    if single:
        return out[0]
    return out
