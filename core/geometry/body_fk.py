"""SOMA FK 前向运动学。"""

from __future__ import annotations

import numpy as np

from core.geometry.rotations import rodrigues
from core.geometry.soma_skeleton import SomaSkeleton


def fk_forward(
    pose: np.ndarray,
    trans: np.ndarray,
    skeleton: SomaSkeleton,
) -> tuple[np.ndarray, np.ndarray]:
    """pose (77,3) 轴角；trans (3,) 根位移 → joints (78,3), world_rots (78,3,3)。"""
    pose = np.asarray(pose, dtype=np.float64).reshape(77, 3)
    trans = np.asarray(trans, dtype=np.float64).reshape(3)
    n = skeleton.num_joints
    joints = np.zeros((n, 3), dtype=np.float64)
    world_rots = np.zeros((n, 3, 3), dtype=np.float64)
    joints[0] = trans
    world_rots[0] = np.eye(3)
    rest = skeleton.rest_local.astype(np.float64)
    parent = skeleton.parent
    for j in range(1, n):
        p = int(parent[j])
        local_rot = rodrigues(pose[j - 1])
        world_rots[j] = world_rots[p] @ local_rot
        joints[j] = joints[p] + world_rots[p] @ rest[j]
    return joints.astype(np.float32), world_rots.astype(np.float64)


def coco_joints_from_fk(
    joints: np.ndarray,
    skeleton: SomaSkeleton,
) -> np.ndarray:
    """78 关节 → 17 COCO 3D（含 offset）。"""
    idx = skeleton.coco_to_soma
    out = joints[idx].astype(np.float32) + skeleton.coco_offset.astype(np.float32)
    return out


def pelvis_alignment_delta(
    joints: np.ndarray,
    coco3d: np.ndarray,
    skeleton: SomaSkeleton,
) -> np.ndarray:
    """SOMA 髋窝 (LeftLeg/RightLeg) 中心 → COCO 髋中点，修正 Root/Hips 语义偏差。"""
    coco = np.asarray(coco3d, dtype=np.float64).reshape(-1, 4)
    j = np.asarray(joints, dtype=np.float64).reshape(-1, 3)
    if coco.shape[0] < 13:
        return np.zeros(3, dtype=np.float64)
    lh, rh = 11, 12
    if not (
        np.all(np.isfinite(coco[lh, :3]))
        and np.all(np.isfinite(coco[rh, :3]))
    ):
        return np.zeros(3, dtype=np.float64)
    coco_hip = 0.5 * (coco[lh, :3] + coco[rh, :3])
    j_l = int(skeleton.coco_to_soma[lh])
    j_r = int(skeleton.coco_to_soma[rh])
    if j_l >= j.shape[0] or j_r >= j.shape[0]:
        return np.zeros(3, dtype=np.float64)
    soma_hip = 0.5 * (j[j_l, :3] + j[j_r, :3])
    return (coco_hip - soma_hip).astype(np.float64)


def fk_forward_aligned(
    pose: np.ndarray,
    trans: np.ndarray,
    coco3d: np.ndarray,
    skeleton: SomaSkeleton,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """FK + 骨盆平移对齐 tri COCO 髋。"""
    joints, world_rots = fk_forward(pose, trans, skeleton)
    delta = pelvis_alignment_delta(joints, coco3d, skeleton)
    if float(np.linalg.norm(delta)) > 1e-6:
        trans = np.asarray(trans, dtype=np.float32).reshape(3) + delta.astype(np.float32)
        joints = joints + delta.astype(np.float32)
    return joints, world_rots, trans
