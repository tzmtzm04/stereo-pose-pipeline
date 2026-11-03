"""Anipose 官方 2D/3D medfilt 滤波（adapted from anipose/filter_pose.py, filter_3d.py）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal
from scipy.interpolate import splev, splrep


@dataclass(frozen=True)
class AniposeFilter2DConfig:
    """anipose DEFAULT_CONFIG['filter']（medfilt）。"""

    enabled: bool = True
    medfilt: int = 13
    offset_threshold: float = 25.0
    score_threshold: float = 0.05
    spline: bool = True
    min_valid_frac: float = 0.5
    min_valid_count: int = 5
    max_interp_gap: int = 0


@dataclass(frozen=True)
class AniposeFilter3DConfig:
    """anipose DEFAULT_CONFIG['filter3d']。"""

    enabled: bool = True
    medfilt: int = 17
    offset_threshold: float = 15.0
    # medfilt 后：mid-hip 深度 Z 零相位 Butterworth（SBS 深度跳变）
    hip_z_lowpass: bool = True
    hip_z_cutoff_hz: float = 6.0
    hip_z_order: int = 4
    hip_z_fps: float = 60.0


_COCO_HIP_LEFT = 11
_COCO_HIP_RIGHT = 12
_Z_AXIS = 2


def _nan_helper(y: np.ndarray) -> tuple[np.ndarray, object]:
    return np.isnan(y), lambda z: z.nonzero()[0]


def medfilt_data(values: np.ndarray, size: int = 15) -> np.ndarray:
    padsize = size + 5
    vpad = np.pad(values, (padsize, padsize), mode="median", stat_length=5)
    vpadf = signal.medfilt(vpad, kernel_size=size)
    return vpadf[padsize:-padsize]


def interpolate_data(vals: np.ndarray) -> np.ndarray:
    nans, ix = _nan_helper(vals)
    out = np.copy(vals)
    if np.mean(nans) > 0.85:
        return out
    if np.sum(~nans) < 2:
        return out
    out[nans] = np.interp(ix(nans), ix(~nans), vals[~nans])
    return out


def filter_pose2d_medfilt(
    points: np.ndarray,
    scores: np.ndarray,
    cfg: AniposeFilter2DConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """N×J×2 + N×J scores → 滤波后 2D（单相机，官方 medfilt+spline）。"""
    points = np.asarray(points, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    n_frames, n_joints, _ = points.shape
    out = np.full((n_frames, n_joints, 2), np.nan, dtype=np.float64)
    out_scores = scores.copy()

    k = int(cfg.medfilt)
    if k % 2 == 0:
        k += 1

    for ji in range(n_joints):
        x = np.array(points[:, ji, 0], dtype=np.float64)
        y = np.array(points[:, ji, 1], dtype=np.float64)

        x_for_med = x.copy()
        y_for_med = y.copy()
        x_for_med[~np.isfinite(x_for_med)] = np.nanmedian(x) if np.any(np.isfinite(x)) else 0.0
        y_for_med[~np.isfinite(y_for_med)] = np.nanmedian(y) if np.any(np.isfinite(y)) else 0.0

        xmed = signal.medfilt(x_for_med, kernel_size=k)
        ymed = signal.medfilt(y_for_med, kernel_size=k)

        score = scores[:, ji]
        err = np.abs(x - xmed) + np.abs(y - ymed)
        bad = np.zeros(n_frames, dtype=bool)
        bad[err >= float(cfg.offset_threshold)] = True
        bad[score < float(cfg.score_threshold)] = True
        bad[~np.isfinite(x) | ~np.isfinite(y)] = True

        xy = np.stack([x, y], axis=1)
        xy[bad] = np.nan
        xy_i = np.copy(xy)

        for axis in range(2):
            vals = xy_i[:, axis]
            nans_bool = ~np.isfinite(vals)
            if not np.any(nans_bool):
                xy_i[:, axis] = vals
                continue
            ok_bool = ~nans_bool
            min_frac = float(cfg.min_valid_frac)
            min_count = int(cfg.min_valid_count)
            if np.mean(ok_bool) <= min_frac or np.sum(ok_bool) <= min_count:
                xy_i[:, axis] = vals
                continue

            to_fill = nans_bool.copy()
            max_gap = int(cfg.max_interp_gap)
            if max_gap > 0:
                idx = 0
                while idx < n_frames:
                    if not nans_bool[idx]:
                        idx += 1
                        continue
                    start = idx
                    while idx < n_frames and nans_bool[idx]:
                        idx += 1
                    if idx - start > max_gap:
                        to_fill[start:idx] = False

            ok_ix = np.flatnonzero(ok_bool)
            if ok_ix.size < 2:
                xy_i[:, axis] = vals
                continue
            first_ok, last_ok = int(ok_ix[0]), int(ok_ix[-1])
            # 只填有效段内的洞，避免 splrep/splev 在序列两端外推爆炸
            in_span = (np.arange(n_frames) >= first_ok) & (np.arange(n_frames) <= last_ok)
            to_fill &= in_span
            fill_ix = np.flatnonzero(to_fill)
            if fill_ix.size == 0:
                xy_i[:, axis] = vals
                continue

            if cfg.spline:
                spline = splrep(ok_ix, vals[ok_bool], k=3, s=0)
                vals[fill_ix] = splev(fill_ix, spline)
            else:
                vals[fill_ix] = np.interp(fill_ix, ok_ix, vals[ok_bool])
            xy_i[:, axis] = vals

        out[:, ji] = xy_i

    return out, out_scores


def filter_pose2d_stereo(
    points: np.ndarray,
    scores: np.ndarray,
    cfg: AniposeFilter2DConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """C×N×J×2 → 逐相机 2D medfilt。"""
    if not cfg.enabled:
        return points, scores
    out = np.asarray(points, dtype=np.float64).copy()
    out_scores = np.asarray(scores, dtype=np.float64).copy()
    for ci in range(out.shape[0]):
        pts_f, sc_f = filter_pose2d_medfilt(out[ci], out_scores[ci], cfg)
        out[ci] = pts_f
        out_scores[ci] = sc_f
    return out, out_scores


def reproj_error_per_joint(
    camera_group,
    points: np.ndarray,
    p3ds: np.ndarray,
) -> np.ndarray:
    """C×N×J×2 + N×J×3 → N×J 平均重投影误差（px）。"""
    n_cams, n_frames, n_joints, _ = points.shape
    p3d_flat = p3ds.reshape(n_frames * n_joints, 3)
    p2d_flat = points.reshape(n_cams, n_frames * n_joints, 2)
    err = camera_group.reprojection_error(p3d_flat, p2d_flat, mean=True)
    err = np.asarray(err, dtype=np.float64).reshape(n_frames, n_joints)
    err[~np.isfinite(err)] = 1.0e5
    return err


def _global_joint_local_index(
    joint_indices: tuple[int, ...] | list[int],
    global_idx: int,
) -> int | None:
    try:
        return list(joint_indices).index(int(global_idx))
    except ValueError:
        return None


def butterworth_lowpass_1d(
    values: np.ndarray,
    *,
    fps: float,
    cutoff_hz: float,
    order: int = 4,
) -> np.ndarray:
    """零相位 Butterworth；按连续有效段分别 filtfilt（Pose2Sim 同构）。"""
    vals = np.asarray(values, dtype=np.float64).copy()
    finite = np.isfinite(vals)
    if vals.size == 0 or not np.any(finite):
        return vals

    nyq = max(float(fps), 1.0) / 2.0
    cutoff = min(float(cutoff_hz), nyq * 0.99)
    if cutoff <= 0.0:
        return vals

    ord2 = max(1, int(order) // 2)
    b, a = signal.butter(ord2, cutoff / nyq, btype="low", analog=False)
    padlen = 3 * max(len(a), len(b))

    out = vals.copy()
    idx = np.flatnonzero(finite)
    gaps = np.where(np.diff(idx) > 1)[0] + 1
    for seg in np.split(idx, gaps):
        if len(seg) <= padlen:
            continue
        out[seg] = signal.filtfilt(b, a, vals[seg])
    return out


def filter_hip_pelvis_z_lowpass(
    p3ds: np.ndarray,
    joint_indices: tuple[int, ...] | list[int],
    cfg: AniposeFilter3DConfig,
) -> np.ndarray:
    """medfilt 后：mid-hip Z 低通，左右髋同步平移 ΔZ（保持髋间距）。"""
    if not cfg.hip_z_lowpass:
        return p3ds

    out = np.asarray(p3ds, dtype=np.float64).copy()
    ji_l = _global_joint_local_index(joint_indices, _COCO_HIP_LEFT)
    ji_r = _global_joint_local_index(joint_indices, _COCO_HIP_RIGHT)
    if ji_l is None and ji_r is None:
        return out

    n_frames = out.shape[0]
    z_l = (
        out[:, ji_l, _Z_AXIS].copy()
        if ji_l is not None
        else np.full(n_frames, np.nan, dtype=np.float64)
    )
    z_r = (
        out[:, ji_r, _Z_AXIS].copy()
        if ji_r is not None
        else np.full(n_frames, np.nan, dtype=np.float64)
    )

    mid = np.full(n_frames, np.nan, dtype=np.float64)
    both = np.isfinite(z_l) & np.isfinite(z_r)
    mid[both] = 0.5 * (z_l[both] + z_r[both])
    mid[np.isfinite(z_l) & ~np.isfinite(z_r)] = z_l[
        np.isfinite(z_l) & ~np.isfinite(z_r)
    ]
    mid[np.isfinite(z_r) & ~np.isfinite(z_l)] = z_r[
        np.isfinite(z_r) & ~np.isfinite(z_l)
    ]

    mid_f = butterworth_lowpass_1d(
        mid,
        fps=float(cfg.hip_z_fps),
        cutoff_hz=float(cfg.hip_z_cutoff_hz),
        order=int(cfg.hip_z_order),
    )
    delta = mid_f - mid
    delta[~np.isfinite(mid) | ~np.isfinite(mid_f)] = 0.0

    for ji in (ji_l, ji_r):
        if ji is None:
            continue
        mask = np.isfinite(out[:, ji, _Z_AXIS])
        out[mask, ji, _Z_AXIS] = out[mask, ji, _Z_AXIS] + delta[mask]
    return out


def filter_pose3d_medfilt(
    p3ds: np.ndarray,
    reproj_err: np.ndarray,
    cfg: AniposeFilter3DConfig,
    *,
    joint_indices: tuple[int, ...] | list[int] | None = None,
) -> np.ndarray:
    """N×J×3 + N×J reproj error → 3D medfilt（官方 filter3d）。"""
    out = np.asarray(p3ds, dtype=np.float64).copy()
    n_frames, n_joints, _ = out.shape
    k = int(cfg.medfilt)
    if k % 2 == 0:
        k += 1

    for ji in range(n_joints):
        bad = reproj_err[:, ji] > float(cfg.offset_threshold)
        for axis in range(3):
            values = out[:, ji, axis].copy()
            values[bad] = np.nan
            values_intp = interpolate_data(values)
            out[:, ji, axis] = medfilt_data(values_intp, size=k)

    if joint_indices is not None:
        out = filter_hip_pelvis_z_lowpass(out, joint_indices, cfg)
    return out
