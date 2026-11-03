"""COCO-17 三角化 → Human3.6M-17 布局与 pelvis 居中 coordinate。"""

from __future__ import annotations

import numpy as np

# Human3.6M 17 关节（与 VideoPose3D / BioModule 常用约定一致）
H36M_JOINT_NAMES = (
    "pelvis",
    "r_hip",
    "r_knee",
    "r_ankle",
    "l_hip",
    "l_knee",
    "l_ankle",
    "spine",
    "thorax",
    "head",
    "l_shoulder",
    "l_elbow",
    "l_wrist",
    "r_shoulder",
    "r_elbow",
    "r_wrist",
)

H36M_SKELETON_17 = [
    (0, 1), (1, 2), (2, 3),
    (0, 4), (4, 5), (5, 6),
    (0, 7), (7, 8), (8, 9),
    (8, 10), (10, 11), (11, 12),
    (8, 13), (13, 14), (14, 15),
]

# COCO 索引 → H36M 索引（直接拷贝的关节）
_COCO_TO_H36M = {
    12: 1,
    14: 2,
    16: 3,
    11: 4,
    13: 5,
    15: 6,
    0: 9,
    5: 10,
    7: 11,
    9: 12,
    6: 13,
    8: 14,
    10: 15,
}

_MIN_CONF = 1e-4


def _joint_valid(kpts: np.ndarray, idx: int) -> bool:
    if idx < 0 or idx >= kpts.shape[0]:
        return False
    p = kpts[idx]
    if not np.all(np.isfinite(p[:3])):
        return False
    if float(p[3]) < _MIN_CONF:
        return False
    return float(np.linalg.norm(p[:3])) > 1e-4


def _blend(
    kpts: np.ndarray,
    i: int,
    j: int,
) -> tuple[np.ndarray, float, bool]:
    vi = _joint_valid(kpts, i)
    vj = _joint_valid(kpts, j)
    if vi and vj:
        xyz = (kpts[i, :3] + kpts[j, :3]) * 0.5
        conf = min(float(kpts[i, 3]), float(kpts[j, 3]))
        return xyz.astype(np.float32), conf, True
    if vi:
        return kpts[i, :3].astype(np.float32), float(kpts[i, 3]), True
    if vj:
        return kpts[j, :3].astype(np.float32), float(kpts[j, 3]), True
    return np.zeros(3, dtype=np.float32), 0.0, False


def coco_kpts3d_to_h36m(kpts3d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    COCO (17,4) 左相机米制 → H36M (17,4)。

    返回 (h36m_kpts, valid_mask)。
    """
    k = np.asarray(kpts3d, dtype=np.float32)
    if k.ndim != 2 or k.shape[0] < 17 or k.shape[1] < 4:
        raise ValueError(f"期望 kpts3d (17,4)，得到 {k.shape}")

    out = np.zeros((17, 4), dtype=np.float32)
    valid = np.zeros(17, dtype=bool)

    for coco_i, h36m_i in _COCO_TO_H36M.items():
        if _joint_valid(k, coco_i):
            out[h36m_i, :3] = k[coco_i, :3]
            out[h36m_i, 3] = k[coco_i, 3]
            valid[h36m_i] = True

    pelvis_xyz, pelvis_conf, pelvis_ok = _blend(k, 11, 12)
    if pelvis_ok:
        out[0, :3] = pelvis_xyz
        out[0, 3] = pelvis_conf
        valid[0] = True

    thorax_xyz, thorax_conf, thorax_ok = _blend(k, 5, 6)
    if thorax_ok:
        out[8, :3] = thorax_xyz
        out[8, 3] = thorax_conf
        valid[8] = True

    if valid[0] and valid[8]:
        out[7, :3] = (out[0, :3] + out[8, :3]) * 0.5
        out[7, 3] = min(float(out[0, 3]), float(out[8, 3]))
        valid[7] = True
    elif valid[0]:
        out[7, :3] = out[0, :3]
        out[7, 3] = out[0, 3]
        valid[7] = True
    elif valid[8]:
        out[7, :3] = out[8, :3]
        out[7, 3] = out[8, 3]
        valid[7] = True

    return out, valid


def pelvis_center_h36m(h36m_kpts: np.ndarray) -> np.ndarray:
    """H36M (17,4) → pelvis 居中的 coordinate (17,3)，单位米。"""
    k = np.asarray(h36m_kpts, dtype=np.float32)
    valid = (k[:, 3] >= _MIN_CONF) & (
        np.linalg.norm(k[:, :3], axis=1) > 1e-4
    )
    if float(k[0, 3]) >= _MIN_CONF and np.linalg.norm(k[0, :3]) > 1e-4:
        origin = k[0, :3]
    else:
        hips = []
        for idx in (1, 4):
            if float(k[idx, 3]) >= _MIN_CONF:
                hips.append(k[idx, :3])
        origin = np.mean(hips, axis=0) if hips else np.zeros(3, dtype=np.float32)

    coord = np.zeros((k.shape[0], 3), dtype=np.float32)
    for i in range(k.shape[0]):
        if valid[i]:
            coord[i] = k[i, :3] - origin
    return coord


def h36m_coord_for_export(h36m_kpts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """导出用 coordinate (17,3) + valid (17,)。"""
    coord = pelvis_center_h36m(h36m_kpts)
    valid = (h36m_kpts[:, 3] >= _MIN_CONF) & (
        np.linalg.norm(h36m_kpts[:, :3], axis=1) > 1e-4
    )
    return coord, valid
