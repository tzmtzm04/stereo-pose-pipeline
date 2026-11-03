"""Anipose 官方 Viterbi 2D 轨迹选择（adapted from anipose/filter_pose.py）。

对每关节：在当前帧及前 n_back 帧的候选点中，用 Viterbi 选最连贯路径。
ViTPose 每帧仅 1 个点时，候选来自时序 lookback；若将来有多 hypothesis（N×P），
同一帧 P 个候选一并参与。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


@dataclass(frozen=True)
class ViterbiPose2DConfig:
    """anipose DEFAULT_CONFIG['filter'] 中 viterbi 相关项。"""

    enabled: bool = True
    n_back: int = 5
    thres_dist: float = 25.0
    score_threshold: float = 0.25
    dup_thres: float = 5.0


def remove_dups(pts: np.ndarray, thres: float = 7.0) -> np.ndarray:
    """同帧内距离过近的重复候选置 NaN。"""
    pts = np.asarray(pts, dtype=np.float64)
    if pts.ndim != 3:
        raise ValueError(f"remove_dups 期望 N×P×2，got {pts.shape}")
    n_frames, n_possible, _ = pts.shape
    tindex = (
        np.repeat(np.arange(n_frames)[:, None], n_possible, axis=1) * 100
    )
    pts_ix = np.dstack([pts, tindex])
    pts_flat = pts_ix.reshape(-1, 3)
    check = np.all(np.isfinite(pts_flat), axis=1)
    ix_good = np.where(check)[0]
    if ix_good.size == 0:
        return pts

    tree = cKDTree(pts_flat[ix_good])
    pairs = tree.query_pairs(float(thres))
    if len(pairs) == 0:
        return pts

    shape = (n_frames, n_possible)
    indices = [ix_good[b] for _a, b in pairs]
    i0, i1 = np.unravel_index(indices, shape)
    pts_out = np.copy(pts)
    pts_out[i0, i1] = np.nan
    return pts_out


def viterbi_path(
    points: np.ndarray,
    scores: np.ndarray,
    *,
    n_back: int = 3,
    thres_dist: float = 30.0,
    dup_thres: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """N×P×2 + N×P scores → N×2 轨迹 + N scores（官方 filter_pose.viterbi_path）。"""
    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if points.ndim != 3:
        raise ValueError(f"viterbi_path 期望 N×P×2，got {points.shape}")
    n_frames, n_possible, _ = points.shape
    if scores.shape != (n_frames, n_possible):
        raise ValueError(
            f"scores 形状须为 {n_frames}×{n_possible}，got {scores.shape}"
        )

    points_nans = remove_dups(points, thres=float(dup_thres))

    num_points = np.sum(~np.isnan(points_nans[:, :, 0]), axis=1)
    num_max = int(np.max(num_points)) if n_frames else 0
    if num_max <= 0:
        out_xy = np.full((n_frames, 2), np.nan, dtype=np.float64)
        out_sc = np.zeros(n_frames, dtype=np.float64)
        return out_xy, out_sc

    particles = np.zeros((n_frames, num_max * n_back + 1, 3), dtype=np.float64)
    valid = np.zeros(n_frames, dtype=np.int64)
    for i in range(n_frames):
        s = 0
        for j in range(int(n_back)):
            if i - j < 0:
                break
            ixs = np.where(~np.isnan(points_nans[i - j, :, 0]))[0]
            n_valid = len(ixs)
            particles[i, s : s + n_valid, :2] = points[i - j, ixs]
            particles[i, s : s + n_valid, 2] = scores[i - j, ixs] * np.power(
                2.0, -j
            )
            s += n_valid
        if s == 0:
            particles[i, 0] = [-1.0, -1.0, 0.001]
            s = 1
        valid[i] = s

    n_particles = int(np.max(valid))
    t_logprob = np.full((n_frames, n_particles), -np.inf, dtype=np.float64)
    t_back = np.zeros((n_frames, n_particles), dtype=np.int64)

    t_logprob[0, : valid[0]] = np.log(particles[0, : valid[0], 2])
    t_back[0, :] = -1

    for i in range(1, n_frames):
        va, vb = int(valid[i - 1]), int(valid[i])
        pa = particles[i - 1, :va, :2]
        pb = particles[i, :vb, :2]

        dists = cdist(pa, pb)
        cdf_high = stats.norm.logcdf(dists + 2.0, scale=float(thres_dist))
        cdf_low = stats.norm.logcdf(dists - 2.0, scale=float(thres_dist))
        cdfs = np.array([cdf_high, cdf_low])
        p_trans = logsumexp(cdfs.T, b=[1, -1], axis=2)
        p_trans[p_trans < -100] = -100.0
        p_trans[pb[:, 0] == -1, :] = np.log(0.001)
        p_trans[:, pa[:, 0] == -1] = np.log(0.001)

        pflat = particles[i, :vb, 2]
        possible = t_logprob[i - 1, :va] + p_trans
        t_logprob[i, :vb] = np.max(possible, axis=1) + np.log(pflat)
        t_back[i, :vb] = np.argmax(possible, axis=1)

    out = np.zeros(n_frames, dtype=np.int64)
    out[-1] = int(np.argmax(t_logprob[-1]))
    for i in range(n_frames - 1, 0, -1):
        out[i - 1] = t_back[i, out[i]]

    trace = np.array([particles[i, out[i]] for i in range(n_frames)])
    return trace[:, :2], trace[:, 2]


def _as_possible_stack(
    points: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """N×J×2 / N×J×P×2 → 逐关节 N×P×2。"""
    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    if points.ndim == 3:
        pts = points[:, :, None, :]
        scs = scores[:, :, None]
    elif points.ndim == 4:
        pts = points
        scs = scores
    else:
        raise ValueError(f"points 维度不支持: {points.shape}")
    return pts, scs


def filter_pose2d_viterbi(
    points: np.ndarray,
    scores: np.ndarray,
    cfg: ViterbiPose2DConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """单相机：N×J×(×P)×2 → Viterbi 平滑轨迹。"""
    if not cfg.enabled:
        return points, scores

    pts4, scs3 = _as_possible_stack(points, scores)
    n_frames, n_joints, n_possible, _ = pts4.shape
    out = np.full((n_frames, n_joints, 2), np.nan, dtype=np.float64)
    out_scores = np.zeros((n_frames, n_joints), dtype=np.float64)

    pts4 = pts4.copy()
    scs3 = scs3.copy()
    low = float(cfg.score_threshold)
    pts4[scs3 < low] = np.nan

    for ji in range(n_joints):
        pts_new, scs_new = viterbi_path(
            pts4[:, ji, :, :],
            scs3[:, ji, :],
            n_back=int(cfg.n_back),
            thres_dist=float(cfg.thres_dist),
            dup_thres=float(cfg.dup_thres),
        )
        out[:, ji] = pts_new
        out_scores[:, ji] = scs_new

    return out, out_scores


def filter_pose2d_viterbi_stereo(
    points: np.ndarray,
    scores: np.ndarray,
    cfg: ViterbiPose2DConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """C×N×J×(×P)×2 → 逐相机 Viterbi，输出 C×N×J×2。"""
    if not cfg.enabled:
        return points, scores
    pts_in = np.asarray(points, dtype=np.float64)
    sc_in = np.asarray(scores, dtype=np.float64)
    n_cams, n_frames, n_joints = pts_in.shape[0], pts_in.shape[1], pts_in.shape[2]
    out = np.full((n_cams, n_frames, n_joints, 2), np.nan, dtype=np.float64)
    out_scores = np.zeros((n_cams, n_frames, n_joints), dtype=np.float64)
    for ci in range(n_cams):
        pts_f, sc_f = filter_pose2d_viterbi(pts_in[ci], sc_in[ci], cfg)
        out[ci] = pts_f
        out_scores[ci] = sc_f
    return out, out_scores
