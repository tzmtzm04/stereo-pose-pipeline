"""OpenCap PreAugmentation：20 个 OpenPose-style marker（含 midHip）。"""

from __future__ import annotations

import numpy as np

from core.offline.two_stage.coco_wholebody_layout import NUM_KEYPOINTS_BODY_FEET

# OpenCap / OpenPose BODY_25 子集（无面部 5 点），LSTM feature 并集 + midHip root
OPENCAP20_MARKER_NAMES: tuple[str, ...] = (
    "Neck",
    "RShoulder",
    "LShoulder",
    "RHip",
    "LHip",
    "midHip",
    "RKnee",
    "LKnee",
    "RAnkle",
    "LAnkle",
    "RHeel",
    "LHeel",
    "RSmallToe",
    "LSmallToe",
    "RBigToe",
    "LBigToe",
    "RElbow",
    "LElbow",
    "RWrist",
    "LWrist",
)

NUM_OPENCAP20 = len(OPENCAP20_MARKER_NAMES)

# WholeBody body 索引（COCO-17 一致）
_WB = {
    "LShoulder": 5,
    "RShoulder": 6,
    "LElbow": 7,
    "RElbow": 8,
    "LWrist": 9,
    "RWrist": 10,
    "LHip": 11,
    "RHip": 12,
    "LKnee": 13,
    "RKnee": 14,
    "LAnkle": 15,
    "RAnkle": 16,
    "LBigToe": 17,
    "LSmallToe": 18,
    "LHeel": 19,
    "RBigToe": 20,
    "RSmallToe": 21,
    "RHeel": 22,
}

# LSTM lower / upper feature（不含 midHip；root 单独用 midHip）
OPENCAP_LSTM_LOWER_FEATURES: tuple[str, ...] = (
    "Neck",
    "RShoulder",
    "LShoulder",
    "RHip",
    "LHip",
    "RKnee",
    "LKnee",
    "RAnkle",
    "LAnkle",
    "RHeel",
    "LHeel",
    "RSmallToe",
    "LSmallToe",
    "RBigToe",
    "LBigToe",
)

OPENCAP_LSTM_UPPER_FEATURES: tuple[str, ...] = (
    "Neck",
    "RShoulder",
    "LShoulder",
    "RElbow",
    "LElbow",
    "RWrist",
    "LWrist",
)

OPENCAP_LSTM_LOWER_RESPONSE: tuple[str, ...] = (
    "r.ASIS_study",
    "L.ASIS_study",
    "r.PSIS_study",
    "L.PSIS_study",
    "r_knee_study",
    "r_mknee_study",
    "r_ankle_study",
    "r_mankle_study",
    "r_toe_study",
    "r_5meta_study",
    "r_calc_study",
    "L_knee_study",
    "L_mknee_study",
    "L_ankle_study",
    "L_mankle_study",
    "L_toe_study",
    "L_calc_study",
    "L_5meta_study",
    "r_shoulder_study",
    "L_shoulder_study",
    "C7_study",
    "r_thigh1_study",
    "r_thigh2_study",
    "r_thigh3_study",
    "L_thigh1_study",
    "L_thigh2_study",
    "L_thigh3_study",
    "r_sh1_study",
    "r_sh2_study",
    "r_sh3_study",
    "L_sh1_study",
    "L_sh2_study",
    "L_sh3_study",
    "RHJC_study",
    "LHJC_study",
)

OPENCAP_LSTM_UPPER_RESPONSE: tuple[str, ...] = (
    "r_lelbow_study",
    "r_melbow_study",
    "r_lwrist_study",
    "r_mwrist_study",
    "L_lelbow_study",
    "L_melbow_study",
    "L_lwrist_study",
    "L_mwrist_study",
)


def _midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(4,) + (4,) → (4,)；任一侧无效则全 NaN。"""
    out = np.full(4, np.nan, dtype=np.float32)
    if not (np.all(np.isfinite(a[:3])) and np.all(np.isfinite(b[:3]))):
        return out
    if float(a[3]) <= 0 or float(b[3]) <= 0:
        return out
    if float(np.linalg.norm(a[:3])) <= 1e-4 or float(np.linalg.norm(b[:3])) <= 1e-4:
        return out
    out[:3] = 0.5 * (a[:3] + b[:3])
    out[3] = min(float(a[3]), float(b[3]))
    return out


def wholebody133_to_opencap20(
    kpts3d: np.ndarray,
    *,
    min_conf: float = 1e-4,
) -> np.ndarray:
    """
    WholeBody (N,4) N>=23 → OpenCap-20 (20,4)。

    直映 body+feet；Neck / midHip 由肩、髋中点合成。
    """
    k = np.asarray(kpts3d, dtype=np.float32)
    if k.ndim != 2 or k.shape[1] < 3:
        raise ValueError(f"期望 kpts3d (N,3|4)，收到 {k.shape}")
    if k.shape[0] < NUM_KEYPOINTS_BODY_FEET:
        raise ValueError(
            f"OpenCap-20 需要至少 body+feet={NUM_KEYPOINTS_BODY_FEET} 点，收到 {k.shape[0]}"
        )
    if k.shape[1] == 3:
        conf = np.ones((k.shape[0], 1), dtype=np.float32)
        k = np.concatenate([k, conf], axis=1)

    out = np.full((NUM_OPENCAP20, 4), np.nan, dtype=np.float32)
    name_to_i = {n: i for i, n in enumerate(OPENCAP20_MARKER_NAMES)}

    for name, wi in _WB.items():
        p = k[wi]
        if float(p[3]) < min_conf:
            continue
        if not np.all(np.isfinite(p[:3])):
            continue
        if float(np.linalg.norm(p[:3])) <= 1e-4:
            continue
        out[name_to_i[name]] = p[:4]

    out[name_to_i["Neck"]] = _midpoint(
        out[name_to_i["LShoulder"]], out[name_to_i["RShoulder"]]
    )
    out[name_to_i["midHip"]] = _midpoint(
        out[name_to_i["LHip"]], out[name_to_i["RHip"]]
    )
    return out


def write_opencap20_back_to_wholebody(
    kpts3d: np.ndarray,
    oc20: np.ndarray,
) -> np.ndarray:
    """将滤波后的 OpenCap-20 写回 WholeBody body+feet 对应索引（其余点不动）。"""
    full = np.asarray(kpts3d, dtype=np.float32).copy()
    oc = np.asarray(oc20, dtype=np.float32)
    if full.ndim != 2 or oc.shape[0] < NUM_OPENCAP20:
        return full
    name_to_i = {n: i for i, n in enumerate(OPENCAP20_MARKER_NAMES)}
    for name, wi in _WB.items():
        oi = name_to_i[name]
        if float(oc[oi, 3]) <= 0:
            continue
        if not np.all(np.isfinite(oc[oi, :3])):
            continue
        full[wi] = oc[oi]
    return full
