"""髋部速度曲线：位置平滑 → 差分 → 可选速度再平滑，供离线预览观察。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from core.offline.offline_preview import OfflinePoseRow

_LEFT_HIP = 11
_RIGHT_HIP = 12

# 预设：light → stream 逐级更流线
SMOOTH_LEVEL_PRESETS: dict[str, dict[str, Any]] = {
    "light": {
        "smooth_win": 11,
        "smooth_method": "savgol",
        "poly_order": 3,
        "smooth_passes": 1,
        "vel_smooth_win": 0,
        "vel_smooth_passes": 1,
    },
    "medium": {
        "smooth_win": 21,
        "smooth_method": "savgol",
        "poly_order": 3,
        "smooth_passes": 1,
        "vel_smooth_win": 9,
        "vel_smooth_passes": 1,
    },
    "strong": {
        "smooth_win": 31,
        "smooth_method": "gaussian",
        "poly_order": 2,
        "smooth_passes": 1,
        "vel_smooth_win": 15,
        "vel_smooth_passes": 1,
    },
    "stream": {
        "smooth_win": 61,
        "smooth_method": "gaussian",
        "poly_order": 2,
        "smooth_passes": 3,
        "vel_smooth_win": 31,
        "vel_smooth_passes": 2,
    },
    "ultra": {
        "smooth_win": 81,
        "smooth_method": "gaussian",
        "poly_order": 2,
        "smooth_passes": 3,
        "vel_smooth_win": 41,
        "vel_smooth_passes": 3,
    },
    "max": {
        "smooth_win": 121,
        "smooth_method": "gaussian",
        "poly_order": 2,
        "smooth_passes": 4,
        "vel_smooth_win": 61,
        "vel_smooth_passes": 4,
    },
    "deep": {
        "smooth_win": 151,
        "smooth_method": "gaussian",
        "poly_order": 2,
        "smooth_passes": 5,
        "vel_smooth_win": 81,
        "vel_smooth_passes": 5,
    },
}


@dataclass(frozen=True)
class HipVelocitySeries:
    capture_seqs: np.ndarray
    walk_axis: tuple[float, float]
    hip_x: np.ndarray
    hip_z: np.ndarray
    s_par: np.ndarray
    v_par_raw: np.ndarray
    v_par_smooth: np.ndarray
    a_par_smooth: np.ndarray
    j_par_smooth: np.ndarray
    z_mid: np.ndarray
    vz_raw: np.ndarray
    vz_smooth: np.ndarray
    a_z_smooth: np.ndarray
    j_z_smooth: np.ndarray
    fps: float
    smooth_win: int
    smooth_method: str
    smooth_level: str = ""
    smooth_passes: int = 1
    vel_smooth_win: int = 0
    vel_smooth_passes: int = 1


def resolve_smooth_params(
    *,
    smooth_level: str | None = None,
    smooth_win: int | None = None,
    smooth_method: str | None = None,
    poly_order: int | None = None,
    smooth_passes: int | None = None,
    vel_smooth_win: int | None = None,
    vel_smooth_passes: int | None = None,
) -> dict[str, Any]:
    """合并 preset 与显式参数（显式优先）。"""
    level = (smooth_level or "medium").lower()
    base = dict(SMOOTH_LEVEL_PRESETS.get(level, SMOOTH_LEVEL_PRESETS["medium"]))
    if smooth_win is not None:
        base["smooth_win"] = int(smooth_win)
    if smooth_method is not None:
        base["smooth_method"] = str(smooth_method)
    if poly_order is not None:
        base["poly_order"] = int(poly_order)
    if smooth_passes is not None:
        base["smooth_passes"] = max(1, int(smooth_passes))
    if vel_smooth_win is not None:
        base["vel_smooth_win"] = max(0, int(vel_smooth_win))
    if vel_smooth_passes is not None:
        base["vel_smooth_passes"] = max(1, int(vel_smooth_passes))
    base.setdefault("vel_smooth_passes", 1)
    base["smooth_level"] = level
    return base


def _hip_mid_xyz(row: OfflinePoseRow) -> tuple[float, float, float] | None:
    if not row.tri_ok or row.kpts3d is None:
        return None
    k = row.kpts3d
    if k.shape[0] <= _RIGHT_HIP:
        return None
    pts = k[[_LEFT_HIP, _RIGHT_HIP], :3]
    if not np.isfinite(pts).all():
        return None
    mid = pts.mean(axis=0)
    return float(mid[0]), float(mid[1]), float(mid[2])


def _fill_short_gaps(arr: np.ndarray, *, max_gap: int = 4) -> np.ndarray:
    out = np.array(arr, dtype=np.float64, copy=True)
    n = len(out)
    i = 0
    while i < n:
        if math.isfinite(out[i]):
            i += 1
            continue
        j = i
        while j < n and not math.isfinite(out[j]):
            j += 1
        gap = j - i
        left = float(out[i - 1]) if i > 0 and math.isfinite(out[i - 1]) else float("nan")
        right = float(out[j]) if j < n and math.isfinite(out[j]) else float("nan")
        if gap <= max_gap and math.isfinite(left) and math.isfinite(right):
            for k, t in enumerate(range(i, j)):
                alpha = (k + 1) / (gap + 1)
                out[t] = left + alpha * (right - left)
        i = max(j, i + 1)
    return out


def _interp_fill(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN 段线性插值；返回 (filled, original_valid_mask)。"""
    n = len(arr)
    out = np.array(arr, dtype=np.float64, copy=True)
    mask = np.isfinite(out)
    if mask.sum() < 2:
        return out, mask
    if mask.all():
        return out, mask
    idx = np.flatnonzero(mask)
    invalid = ~mask
    out[invalid] = np.interp(np.flatnonzero(invalid), idx, out[mask])
    return out, mask


def _moving_average(arr: np.ndarray, win: int) -> np.ndarray:
    n = len(arr)
    filled, valid = _interp_fill(arr)
    if n <= 1 or win <= 1:
        return filled
    w = int(win) | 1
    kernel = np.ones(w, dtype=np.float64) / float(w)
    sm = np.convolve(filled, kernel, mode="same")
    sm[~valid] = np.nan
    return sm


def _ema(arr: np.ndarray, alpha: float) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=np.float64)
    prev = float("nan")
    for i, v in enumerate(arr):
        if not math.isfinite(v):
            out[i] = prev
            continue
        if not math.isfinite(prev):
            prev = float(v)
        else:
            prev = alpha * float(v) + (1.0 - alpha) * prev
        out[i] = prev
    return out


def _gaussian(arr: np.ndarray, win: int) -> np.ndarray:
    n = len(arr)
    w = max(3, int(win) | 1)
    if n <= 1:
        return np.array(arr, dtype=np.float64, copy=True)
    filled, valid = _interp_fill(arr)
    sigma = max(w / 5.0, 1.0)
    x = np.arange(w, dtype=np.float64) - (w // 2)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    sm = np.convolve(filled, kernel, mode="same")
    sm[~valid] = np.nan
    return sm


def _savgol(arr: np.ndarray, win: int, poly: int) -> np.ndarray:
    n = len(arr)
    w = max(3, int(win) | 1)
    if n < w:
        return _moving_average(arr, w)
    try:
        from scipy.signal import savgol_filter
    except Exception:
        return _gaussian(arr, w)
    filled, valid = _interp_fill(arr)
    p = min(max(1, poly), w - 2)
    try:
        sm = savgol_filter(filled, window_length=w, polyorder=p, mode="interp")
    except Exception:
        return _gaussian(arr, w)
    sm = np.asarray(sm, dtype=np.float64)
    sm[~valid] = np.nan
    return sm


def smooth_series(
    arr: np.ndarray,
    *,
    win: int,
    method: str = "savgol",
    poly_order: int = 3,
    passes: int = 1,
) -> np.ndarray:
    w = max(3, int(win) | 1)
    m = method.lower()
    out = np.array(arr, dtype=np.float64, copy=True)
    n_pass = max(1, int(passes))
    for _ in range(n_pass):
        if m == "moving_avg":
            out = _moving_average(out, w)
        elif m == "ema":
            alpha = 2.0 / (w + 1.0)
            out = _ema(out, alpha)
        elif m == "gaussian":
            out = _gaussian(out, w)
        else:
            out = _savgol(out, w, poly_order)
    return out


def smooth_position(
    arr: np.ndarray,
    *,
    win: int,
    method: str = "savgol",
    poly_order: int = 3,
    passes: int = 1,
) -> np.ndarray:
    return smooth_series(
        arr,
        win=win,
        method=method,
        poly_order=poly_order,
        passes=passes,
    )


def estimate_walk_axis(
    hip_x: np.ndarray,
    hip_z: np.ndarray,
    *,
    vx: np.ndarray | None = None,
    vz: np.ndarray | None = None,
) -> tuple[float, float]:
    mask = np.isfinite(hip_x) & np.isfinite(hip_z)
    if mask.sum() < 8:
        return 0.0, 1.0
    P = np.stack([hip_x[mask], hip_z[mask]], axis=1)
    P = P - P.mean(axis=0)
    if float(np.max(np.abs(P))) < 1e-4:
        return 0.0, 1.0
    _, _, vt = np.linalg.svd(P, full_matrices=False)
    ux, uz = float(vt[0, 0]), float(vt[0, 1])
    if vx is not None and vz is not None:
        vm = np.isfinite(vx) & np.isfinite(vz) & mask
        if vm.sum() >= 5:
            spd = np.hypot(vx[vm], vz[vm])
            if float(spd.sum()) > 1e-6:
                mean_v = np.array(
                    [
                        float(np.sum(vx[vm] * spd) / spd.sum()),
                        float(np.sum(vz[vm] * spd) / spd.sum()),
                    ]
                )
                if float(np.dot(mean_v, vt[0])) < 0.0:
                    ux, uz = -ux, -uz
    n = math.hypot(ux, uz)
    if n < 1e-6:
        return 0.0, 1.0
    return ux / n, uz / n


def _differentiate(pos: np.ndarray, fps: float) -> np.ndarray:
    n = len(pos)
    v = np.full(n, np.nan, dtype=np.float64)
    dt = 1.0 / max(float(fps), 1e-6)
    for i in range(1, n):
        if math.isfinite(pos[i]) and math.isfinite(pos[i - 1]):
            v[i] = (pos[i] - pos[i - 1]) / dt
    if n > 1 and math.isfinite(pos[1]) and math.isfinite(pos[0]):
        v[0] = (pos[1] - pos[0]) / dt
    return v


def _smooth_velocity(
    v: np.ndarray,
    *,
    win: int,
    method: str = "gaussian",
    passes: int = 1,
) -> np.ndarray:
    if win <= 0:
        return v
    return smooth_series(
        v,
        win=win,
        method=method,
        poly_order=2,
        passes=max(1, int(passes)),
    )


def compute_hip_velocity_series(
    poses: list[OfflinePoseRow],
    fps: float,
    *,
    smooth_level: str | None = "medium",
    smooth_win: int | None = None,
    smooth_method: str | None = None,
    poly_order: int | None = None,
    smooth_passes: int | None = None,
    vel_smooth_win: int | None = None,
    vel_smooth_passes: int | None = None,
    gap_fill: int = 4,
) -> HipVelocitySeries:
    """位置强平滑 → 差分 → 可选速度再平滑。"""
    cfg = resolve_smooth_params(
        smooth_level=smooth_level,
        smooth_win=smooth_win,
        smooth_method=smooth_method,
        poly_order=poly_order,
        smooth_passes=smooth_passes,
        vel_smooth_win=vel_smooth_win,
        vel_smooth_passes=vel_smooth_passes,
    )
    swin = int(cfg["smooth_win"])
    smethod = str(cfg["smooth_method"])
    poly = int(cfg["poly_order"])
    passes = int(cfg["smooth_passes"])
    vwin = int(cfg["vel_smooth_win"])
    vpass = int(cfg.get("vel_smooth_passes", 1))
    level = str(cfg["smooth_level"])

    n = len(poses)
    seqs = np.array([int(p.capture_seq) for p in poses], dtype=np.int64)
    hip_x = np.full(n, np.nan, dtype=np.float64)
    hip_z = np.full(n, np.nan, dtype=np.float64)
    z_mid = np.full(n, np.nan, dtype=np.float64)

    for i, row in enumerate(poses):
        mid = _hip_mid_xyz(row)
        if mid is None:
            continue
        hip_x[i], _, hip_z[i] = mid
        z_mid[i] = mid[2]

    hip_x = _fill_short_gaps(hip_x, max_gap=gap_fill)
    hip_z = _fill_short_gaps(hip_z, max_gap=gap_fill)
    z_mid = _fill_short_gaps(z_mid, max_gap=gap_fill)

    vx_raw = _differentiate(hip_x, fps)
    vz_raw = _differentiate(hip_z, fps)
    axis = estimate_walk_axis(hip_x, hip_z, vx=vx_raw, vz=vz_raw)
    ux, uz = axis
    s_par = hip_x * ux + hip_z * uz

    s_smooth = smooth_position(
        s_par,
        win=swin,
        method=smethod,
        poly_order=poly,
        passes=passes,
    )
    z_smooth = smooth_position(
        z_mid,
        win=swin,
        method=smethod,
        poly_order=poly,
        passes=passes,
    )

    v_par_raw = _differentiate(s_par, fps)
    v_par_smooth = _differentiate(s_smooth, fps)
    vz_smooth = _differentiate(z_smooth, fps)

    if vwin > 0:
        v_par_smooth = _smooth_velocity(
            v_par_smooth, win=vwin, method="gaussian", passes=vpass
        )
        vz_smooth = _smooth_velocity(
            vz_smooth, win=vwin, method="gaussian", passes=vpass
        )

    a_par_smooth = _differentiate(v_par_smooth, fps)
    a_z_smooth = _differentiate(vz_smooth, fps)
    j_par_smooth = _differentiate(a_par_smooth, fps)
    j_z_smooth = _differentiate(a_z_smooth, fps)

    return HipVelocitySeries(
        capture_seqs=seqs,
        walk_axis=axis,
        hip_x=hip_x,
        hip_z=hip_z,
        s_par=s_par,
        v_par_raw=v_par_raw,
        v_par_smooth=v_par_smooth,
        a_par_smooth=a_par_smooth,
        j_par_smooth=j_par_smooth,
        z_mid=z_mid,
        vz_raw=vz_raw,
        vz_smooth=vz_smooth,
        a_z_smooth=a_z_smooth,
        j_z_smooth=j_z_smooth,
        fps=float(fps),
        smooth_win=swin,
        smooth_method=smethod,
        smooth_level=level,
        smooth_passes=passes,
        vel_smooth_win=vwin,
        vel_smooth_passes=vpass,
    )
