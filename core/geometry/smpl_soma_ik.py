"""SMPL-24 关节目标 → SOMA 解析 IK。"""

from __future__ import annotations

import numpy as np

from core.geometry.body_fk import fk_forward
from core.geometry.body_ik import (
    _set_joint_rotation_for_child_bone,
    _swing_ik_chain_to_targets,
)
from core.geometry.coco_smpl_map import smpl24_joints_to_soma_targets
from core.geometry.joint_maps import BodyJointMaps, default_body_joint_maps
from core.geometry.soma_skeleton import SomaSkeleton


def _smpl_point(smpl24: np.ndarray, idx: int, min_conf: float) -> np.ndarray | None:
    s = np.asarray(smpl24, dtype=np.float32).reshape(24, 4)
    if idx < 0 or idx >= 24:
        return None
    if float(s[idx, 3]) < min_conf:
        return None
    xyz = s[idx, :3]
    if not np.all(np.isfinite(xyz)):
        return None
    return xyz.astype(np.float64)


def _swing_limb_from_smpl(
    pose: np.ndarray,
    skeleton: SomaSkeleton,
    chain: list[int],
    smpl24: np.ndarray,
    smpl_triplet: tuple[int, ...],
    trans: np.ndarray,
    *,
    min_conf: float,
) -> None:
    """四肢：SMPL shoulder/elbow/wrist → SOMA 链 child-bone IK。"""
    pts = [_smpl_point(smpl24, si, min_conf) for si in smpl_triplet]
    if any(p is None for p in pts[1:]):
        return
    # SMPL [collar, shoulder, elbow, wrist]；SOMA 4 节链用肩/肘/腕 3 目标
    targets = [pts[1], pts[2], pts[3]]
    _swing_ik_chain_to_targets(pose, skeleton, chain, trans, targets)


def _spine_from_smpl(
    pose: np.ndarray,
    skeleton: SomaSkeleton,
    smpl24: np.ndarray,
    maps: BodyJointMaps,
    trans: np.ndarray,
    *,
    min_conf: float,
) -> None:
    """脊柱：逐段对准 SMPL pelvis→spine→neck→head。"""
    smpl_idx = list(maps.spine_smpl_indices)
    soma_chain = list(maps.spine_soma_indices)
    if len(soma_chain) < 2:
        return

    targets: list[np.ndarray] = []
    for si in smpl_idx[1:]:
        p = _smpl_point(smpl24, si, min_conf)
        if p is not None:
            targets.append(p)
    if len(targets) < 1:
        return

    trans_use = np.asarray(trans, dtype=np.float32).reshape(3)
    joints, world_rots = fk_forward(pose, trans_use, skeleton)
    n_seg = min(len(soma_chain) - 1, len(targets))
    for k in range(n_seg):
        j = soma_chain[k]
        child = soma_chain[k + 1]
        direction = targets[k] - joints[j].astype(np.float64)
        if float(np.linalg.norm(direction)) < 1e-4:
            continue
        p = int(skeleton.parent[j])
        world_rots[j] = _set_joint_rotation_for_child_bone(
            pose,
            skeleton,
            j,
            child,
            world_rots[p],
            direction,
        )
        joints, world_rots = fk_forward(pose, trans_use, skeleton)


def smpl24_ik_init(
    skeleton: SomaSkeleton,
    smpl24: np.ndarray,
    *,
    maps: BodyJointMaps | None = None,
    min_conf: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """SMPL-24 关节位置 → SOMA pose (77,3) + trans (3,)。"""
    maps = maps or default_body_joint_maps()
    smpl = np.asarray(smpl24, dtype=np.float32).reshape(24, 4)
    pose = np.zeros((77, 3), dtype=np.float32)
    trans = np.zeros(3, dtype=np.float32)

    pelvis = _smpl_point(smpl, 0, min_conf)
    if pelvis is not None:
        trans = pelvis.astype(np.float32)
    elif _smpl_point(smpl, 1, min_conf) is not None and _smpl_point(smpl, 2, min_conf) is not None:
        trans = (
            0.5
            * (
                smpl[1, :3].astype(np.float32) + smpl[2, :3].astype(np.float32)
            )
        ).astype(np.float32)

    _spine_from_smpl(pose, skeleton, smpl, maps, trans, min_conf=min_conf)

    for name, smpl_trip in maps.limb_smpl_triplets.items():
        chain = list(maps.limb_soma_chains.get(name, ()))
        if len(chain) < 3:
            continue
        _swing_limb_from_smpl(
            pose,
            skeleton,
            chain,
            smpl,
            smpl_trip,
            trans,
            min_conf=min_conf,
        )

    return pose, trans


def smpl24_usable_for_ik(
    smpl24: np.ndarray,
    *,
    min_conf: float = 0.25,
    min_valid: int = 8,
) -> bool:
    s = np.asarray(smpl24, dtype=np.float32).reshape(24, 4)
    valid = (s[:, 3] >= float(min_conf)) & np.all(np.isfinite(s[:, :3]), axis=1)
    if int(valid.sum()) < int(min_valid):
        return False
    return bool(valid[0] or (valid[1] and valid[2]))


def smpl_targets_mean_error(
    joints78: np.ndarray,
    smpl24: np.ndarray,
    *,
    maps: BodyJointMaps | None = None,
) -> float:
    """FK 关节 vs SMPL 映射目标的平均距离（米）。"""
    maps = maps or default_body_joint_maps()
    targets = smpl24_joints_to_soma_targets(smpl24, maps=maps)
    j = np.asarray(joints78, dtype=np.float64).reshape(-1, 3)
    dists: list[float] = []
    for sj, tgt in targets.items():
        if sj >= j.shape[0]:
            continue
        dists.append(float(np.linalg.norm(j[sj] - tgt)))
    return float(np.mean(dists)) if dists else 0.0
