"""前 N 帧 DLT 三角化 → body_limbs 骨段长度中位数标定。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.geometry.anipose_constraints import BODY_LIMBS_BONE_CONSTRAINTS
from core.geometry.calib_loader import load_stereo_calibration
from core.geometry.triangulate import triangulate_matched_keypoints

# body_limbs 局部下标 → COCO-17 全局下标
BODY_LIMBS_JOINT_INDICES: tuple[int, ...] = tuple(range(5, 17))

BONE_NAMES: tuple[str, ...] = (
    "shoulder_line",
    "left_upper_arm",
    "left_forearm",
    "right_upper_arm",
    "right_forearm",
    "left_shoulder_hip",
    "right_shoulder_hip",
    "hip_line",
    "left_thigh",
    "left_shin",
    "right_thigh",
    "right_shin",
)

# 骨长选峰 SSE：仅刚性肢段（上臂、大腿）；不含肩线/肩髋/髋线（随姿态变）
CORE_BONE_INDICES: tuple[int, ...] = (1, 3, 8, 10)

# body_limbs 12 点中参与多峰搜索的 8 点（肩/肘/髋/膝）；腕踝固定 top-1
# 局部下标：0 L肩 1 R肩 2 L肘 3 R肘 | 4 L腕 5 R腕 | 6 L髋 7 R髋 8 L膝 9 R膝 | 10 L踝 11 R踝
SEARCH_JOINT_LOCAL_INDICES: tuple[int, ...] = (0, 1, 2, 3, 6, 7, 8, 9)
FIXED_JOINT_LOCAL_INDICES: tuple[int, ...] = (4, 5, 10, 11)
N_BODY_LIMBS_JOINTS: int = 12

# 左右对称骨段：同组样本合并后再取中位数
SYMMETRIC_BONE_GROUPS: tuple[tuple[str, ...], ...] = (
    ("shoulder_line",),
    ("left_upper_arm", "right_upper_arm"),
    ("left_forearm", "right_forearm"),
    ("left_shoulder_hip", "right_shoulder_hip"),
    ("hip_line",),
    ("left_thigh", "right_thigh"),
    ("left_shin", "right_shin"),
)

_BONE_NAME_TO_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(BONE_NAMES)
}


def body_limbs_bones_global() -> tuple[tuple[int, int], ...]:
    """12 骨段，端点为 COCO-17 全局索引。"""
    idx = BODY_LIMBS_JOINT_INDICES
    return tuple((int(idx[a]), int(idx[b])) for a, b in BODY_LIMBS_BONE_CONSTRAINTS)


def _load_rows(
    path: Path,
    *,
    start_seq: int | None,
    end_seq: int | None,
) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            seq = int(row.get("capture_seq", -1))
            if start_seq is not None and seq < int(start_seq):
                continue
            if end_seq is not None and seq > int(end_seq):
                continue
            rows.append(row)
    return rows


def _bone_lengths_from_kpts3d(kpts3d: np.ndarray) -> np.ndarray:
    """12 骨段 3D 长度；无效为 nan。"""
    k = np.asarray(kpts3d, dtype=np.float64)
    out = np.full(len(BODY_LIMBS_BONE_CONSTRAINTS), np.nan, dtype=np.float64)
    for bi, (a, b) in enumerate(body_limbs_bones_global()):
        pa = k[a, :3]
        pb = k[b, :3]
        wa = float(k[a, 3]) if k.shape[1] > 3 else 1.0
        wb = float(k[b, 3]) if k.shape[1] > 3 else 1.0
        if wa <= 0 or wb <= 0:
            continue
        if not (np.all(np.isfinite(pa)) and np.all(np.isfinite(pb))):
            continue
        out[bi] = float(np.linalg.norm(pb - pa))
    return out


@dataclass
class BoneCalibStats:
    frames_in: int = 0
    frames_tri_ok: int = 0
    samples_per_bone: tuple[int, ...] = ()
    bone_lengths_m: tuple[float, ...] = ()


def calibrate_bone_lengths_median(
    *,
    pose2d_path: Path,
    calib_path: Path,
    start_seq: int,
    end_seq: int,
    min_conf: float = 0.25,
    stereo_rectified: bool = False,
) -> tuple[dict[str, float], BoneCalibStats, dict]:
    """top-1 2D → DLT → 各骨段长度样本的中位数。"""
    calib = load_stereo_calibration(calib_path)
    rows = _load_rows(pose2d_path, start_seq=start_seq, end_seq=end_seq)
    n_bones = len(BODY_LIMBS_BONE_CONSTRAINTS)
    samples: list[list[float]] = [[] for _ in range(n_bones)]
    tri_ok = 0
    peak_p = None
    peak_frames = 0

    for row in rows:
        if "det_left_kpts_hypo" in row and "det_right_kpts_hypo" in row:
            peak_frames += 1
            lh = np.asarray(row["det_left_kpts_hypo"])
            if peak_p is None and lh.ndim == 3:
                peak_p = int(lh.shape[1])
        kl_raw = row.get("det_left_kpts")
        kr_raw = row.get("det_right_kpts")
        if kl_raw is None or kr_raw is None:
            continue
        kl = np.asarray(kl_raw, dtype=np.float64)
        kr = np.asarray(kr_raw, dtype=np.float64)
        if kl.ndim != 2 or kr.ndim != 2 or kl.shape != kr.shape:
            continue
        kpts3d = triangulate_matched_keypoints(
            kl[:, :2],
            kr[:, :2],
            calib,
            min_conf=float(min_conf),
            conf_left=kl[:, 2],
            conf_right=kr[:, 2],
            mode="opencv",
            stereo_rectified=bool(stereo_rectified),
        )
        if kpts3d is None or float(np.asarray(kpts3d)[:, 3].sum()) <= 0:
            continue
        tri_ok += 1
        bl = _bone_lengths_from_kpts3d(kpts3d)
        for bi, length in enumerate(bl):
            if np.isfinite(length) and length > 0:
                samples[bi].append(float(length))

    # 对称骨段：合并样本后取组内单一中位数
    medians: dict[str, float] = {}
    bone_entries: list[dict] = []
    counts: list[int] = []
    symmetric_groups_out: list[dict] = []

    for group in SYMMETRIC_BONE_GROUPS:
        pooled: list[float] = []
        for name in group:
            pooled.extend(samples[_BONE_NAME_TO_INDEX[name]])
        arr = np.asarray(pooled, dtype=np.float64)
        med = float(np.median(arr)) if arr.size else float("nan")
        group_label = group[0] if len(group) == 1 else "/".join(group)
        symmetric_groups_out.append(
            {
                "names": list(group),
                "length_m_median": med,
                "n_samples_pooled": int(arr.size),
            }
        )
        for name in group:
            bi = _BONE_NAME_TO_INDEX[name]
            per_side = np.asarray(samples[bi], dtype=np.float64)
            counts.append(int(arr.size))  # 组内 pooled 样本数
            medians[name] = med
            a, b = body_limbs_bones_global()[bi]
            bone_entries.append(
                {
                    "name": name,
                    "joints": [a, b],
                    "symmetric_group": group_label,
                    "length_m_median": med,
                    "n_samples_side": int(per_side.size),
                    "n_samples_pooled": int(arr.size),
                    "length_m_mean_pooled": float(np.mean(arr)) if arr.size else None,
                    "length_m_std_pooled": float(np.std(arr)) if arr.size > 1 else None,
                }
            )

    stats = BoneCalibStats(
        frames_in=len(rows),
        frames_tri_ok=tri_ok,
        samples_per_bone=tuple(counts),
        bone_lengths_m=tuple(medians[n] for n in BONE_NAMES),
    )
    meta = {
        "bones": bone_entries,
        "symmetric_pool": True,
        "symmetric_groups": symmetric_groups_out,
        "hypo": {
            "frames_with_hypo": peak_frames,
            "peaks_per_joint": peak_p,
        },
    }
    return medians, stats, meta
