"""3D 关键点分轴 Butterworth 低通（Z 可单独更狠）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.geometry.anipose_filter import butterworth_lowpass_1d, interpolate_data


@dataclass(frozen=True)
class Pose3DAxisLowpassConfig:
    fps: float = 60.0
    z_cutoff_hz: float = 4.0
    xy_cutoff_hz: float = 8.0
    order: int = 4
    min_conf: float = 0.25


_AXIS_CUTOFF = (0, 1, 2)  # x, y, z → cfg.xy, cfg.xy, cfg.z


def _cutoff_for_axis(axis: int, cfg: Pose3DAxisLowpassConfig) -> float:
    if axis == 2:
        return float(cfg.z_cutoff_hz)
    xy = float(cfg.xy_cutoff_hz)
    return xy


def filter_pose3d_axis_lowpass(
    kpts_stack: np.ndarray,
    cfg: Pose3DAxisLowpassConfig,
) -> np.ndarray:
    """N×J×4 (x,y,z,conf) → 分轴零相位低通；低 conf 帧置 nan 再插值。"""
    k = np.asarray(kpts_stack, dtype=np.float64)
    if k.ndim != 3 or k.shape[2] < 4:
        raise ValueError(f"期望 N×J×4， got {k.shape}")

    out = k.copy()
    n_frames, n_joints, _ = out.shape
    min_conf = float(cfg.min_conf)

    for ji in range(n_joints):
        conf = out[:, ji, 3]
        valid = conf >= min_conf
        if int(np.sum(valid)) < 8:
            continue
        for axis in range(3):
            cutoff = _cutoff_for_axis(axis, cfg)
            if cutoff <= 0.0:
                continue
            vals = out[:, ji, axis].copy()
            vals[~valid] = np.nan
            vals = interpolate_data(vals)
            vals_f = butterworth_lowpass_1d(
                vals,
                fps=float(cfg.fps),
                cutoff_hz=cutoff,
                order=int(cfg.order),
            )
            mask = valid & np.isfinite(vals_f)
            out[mask, ji, axis] = vals_f[mask]
    return out.astype(np.float32)


def filter_pose3d_rows_axis_lowpass(
    rows: list[dict],
    cfg: Pose3DAxisLowpassConfig,
) -> list[dict]:
    """按 capture_seq 排序的 jsonl 行 → 滤波后行（仅更新 kpts3d）。"""
    if not rows:
        return []

    by_seq = {int(r.get("capture_seq", i)): r for i, r in enumerate(rows)}
    seqs = sorted(by_seq)
    template = by_seq[seqs[0]]
    raw0 = template.get("kpts3d")
    if raw0 is None:
        return list(rows)
    n_joints = int(np.asarray(raw0).shape[0])

    stack = np.zeros((len(seqs), n_joints, 4), dtype=np.float64)
    for fi, seq in enumerate(seqs):
        raw = by_seq[seq].get("kpts3d")
        if raw is None:
            continue
        k = np.asarray(raw, dtype=np.float64)
        if k.shape[0] != n_joints:
            raise ValueError(f"capture_seq={seq} 关节数 {k.shape[0]} != {n_joints}")
        stack[fi, : k.shape[0], : min(4, k.shape[1])] = k[:, : min(4, k.shape[1])]

    filtered = filter_pose3d_axis_lowpass(stack, cfg)

    out_rows: list[dict] = []
    for fi, seq in enumerate(seqs):
        row = dict(by_seq[seq])
        row["kpts3d"] = filtered[fi].tolist()
        row["axis_lowpass_z"] = True
        out_rows.append(row)
    return out_rows
