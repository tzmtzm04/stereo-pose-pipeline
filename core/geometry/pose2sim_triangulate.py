"""Pose2Sim 加权 DLT + 重投影质检三角化（双目 / 多目）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from core.geometry.stereo_rectify import rectified_to_left_cam_opencv
from core.offline.two_stage.coco_wholebody_layout import skip_wholebody_hand_triangulation
from core.geometry.pose2sim_triangulation_core import triangulation_from_best_cameras

TriangulationMode = Literal["opencv", "pose2sim"]


@dataclass(frozen=True)
class Pose2SimTriangulationConfig:
    reproj_error_threshold_px: float = 15.0
    likelihood_threshold: float = 0.3
    min_cameras_for_triangulation: int = 2
    handle_lr_swap: bool = True
    undistort_points: bool = False


def parse_triangulation_mode(name: str) -> TriangulationMode:
    key = str(name).strip().lower().replace("-", "_")
    if key in ("opencv", "cv", "dlt", "legacy"):
        return "opencv"
    if key in ("pose2sim", "p2s", "default"):
        return "pose2sim"
    raise ValueError(f"未知 triangulation 模式: {name!r}（opencv | pose2sim）")


def build_lr_swap_indices(j_count: int) -> list[int]:
    """COCO-WholeBody 133：左右对称关键点互换索引。"""
    swap = list(range(j_count))
    pairs = [
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
        (9, 10),
        (11, 12),
        (13, 14),
        (15, 16),
        (17, 20),
        (18, 21),
        (19, 22),
    ]
    for a, b in pairs:
        if a < j_count and b < j_count:
            swap[a], swap[b] = b, a
    for k in range(21):
        li, ri = 91 + k, 112 + k
        if li < j_count and ri < j_count:
            swap[li] = ri
            swap[ri] = li
    return swap


def projection_matrices_from_stereo(
    calib: StereoCalibration,
    *,
    stereo_rectified: bool = False,
) -> list[np.ndarray]:
    """投影矩阵：原图路径 K[R|t]；校正路径 P1/P2。"""
    if stereo_rectified:
        p1, p2 = projection_matrices_rectified(calib)
        return [p1, p2]
    mats: list[np.ndarray] = []
    specs = (
        (calib.K1, np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64)),
        (calib.K2, calib.R, calib.T.reshape(3)),
    )
    for k, r, t in specs:
        kh = np.hstack([k, np.zeros((3, 1), dtype=np.float64)])
        h = np.vstack(
            [
                np.hstack([r, t.reshape(3, 1)]),
                np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
            ]
        )
        mats.append(kh @ h)
    return mats


def _pose2sim_config_dict(cfg: Pose2SimTriangulationConfig) -> dict:
    return {
        "triangulation": {
            "reproj_error_threshold_triangulation": float(cfg.reproj_error_threshold_px),
            "likelihood_threshold_triangulation": float(cfg.likelihood_threshold),
            "min_cameras_for_triangulation": int(cfg.min_cameras_for_triangulation),
            "handle_LR_swap": bool(cfg.handle_lr_swap),
            "undistort_points": bool(cfg.undistort_points),
        }
    }


def triangulate_matched_keypoints_pose2sim(
    kpts_left: np.ndarray,
    kpts_right: np.ndarray,
    calib: StereoCalibration,
    *,
    min_conf: float = 0.0,
    conf_left: np.ndarray | None = None,
    conf_right: np.ndarray | None = None,
    cfg: Pose2SimTriangulationConfig | None = None,
    calib_toml: Path | str | None = None,
    stereo_rectified: bool = False,
    exclude_wholebody_hands: bool = True,
) -> np.ndarray:
    """Pose2Sim 逐点三角化，输出形状与 triangulate_matched_keypoints 相同。"""
    _ = calib_toml  # 投影矩阵由 stereo 标定构建，与 Calib.toml 一致
    cfg = cfg or Pose2SimTriangulationConfig()
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
    p_mats = projection_matrices_from_stereo(calib, stereo_rectified=stereo_rectified)
    config_dict = _pose2sim_config_dict(cfg)
    swap_idx = build_lr_swap_indices(j_count)
    like_thresh = max(float(min_conf), float(cfg.likelihood_threshold))

    for pi in range(p_count):
        for ji in range(j_count):
            if skip_wholebody_hand_triangulation(
                ji, j_count, enabled=exclude_wholebody_hands
            ):
                continue
            c_l = float(conf_left[pi, ji]) if conf_left is not None else 1.0
            c_r = float(conf_right[pi, ji]) if conf_right is not None else 1.0
            if min(c_l, c_r) < like_thresh:
                continue
            pl = kl[pi, ji]
            pr = kr[pi, ji]
            if pl[0] <= 0 or pr[0] <= 0:
                continue

            x_cams = np.array([float(pl[0]), float(pr[0])], dtype=np.float64)
            y_cams = np.array([float(pl[1]), float(pr[1])], dtype=np.float64)
            conf_cams = np.array([c_l, c_r], dtype=np.float64)

            sj = swap_idx[ji] if ji < len(swap_idx) else ji
            pls = kl[pi, sj]
            prs = kr[pi, sj]
            c_ls = float(conf_left[pi, sj]) if conf_left is not None else 1.0
            c_rs = float(conf_right[pi, sj]) if conf_right is not None else 1.0
            x_swapped = np.array([float(pls[0]), float(prs[0])], dtype=np.float64)
            y_swapped = np.array([float(pls[1]), float(prs[1])], dtype=np.float64)
            conf_swapped = np.array([c_ls, c_rs], dtype=np.float64)

            q, _err, _nb_excl, _id_excl = triangulation_from_best_cameras(
                config_dict,
                (x_cams, y_cams, conf_cams),
                (x_swapped, y_swapped, conf_swapped),
                p_mats,
                None,
            )
            if not np.all(np.isfinite(q)):
                continue
            if stereo_rectified:
                q = rectified_to_left_cam_opencv(q, calib.R1)
                q = q / 1000.0
            out[pi, ji, 0:3] = q.astype(np.float32)
            out[pi, ji, 3] = min(c_l, c_r)

    if single:
        return out[0]
    return out
