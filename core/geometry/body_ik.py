"""解析 IK：HybrIK Swing（四肢）+ FABRIK（脊柱）。"""

from __future__ import annotations

import numpy as np

from core.geometry.body_fk import fk_forward
from core.geometry.rotations import axis_angle_from_matrix, normalize, rotation_between
from core.geometry.soma_skeleton import SomaSkeleton

# COCO 17 索引
COCO = {
    "nose": 0,
    "l_shoulder": 5,
    "r_shoulder": 6,
    "l_elbow": 7,
    "r_elbow": 8,
    "l_wrist": 9,
    "r_wrist": 10,
    "l_hip": 11,
    "r_hip": 12,
    "l_knee": 13,
    "r_knee": 14,
    "l_ankle": 15,
    "r_ankle": 16,
}

_LIMB_COCO = (
    ("left_arm", COCO["l_shoulder"], COCO["l_elbow"], COCO["l_wrist"]),
    ("right_arm", COCO["r_shoulder"], COCO["r_elbow"], COCO["r_wrist"]),
    ("left_leg", COCO["l_hip"], COCO["l_knee"], COCO["l_ankle"]),
    ("right_leg", COCO["r_hip"], COCO["r_knee"], COCO["r_ankle"]),
)


def _valid_coco_point(coco3d: np.ndarray, idx: int, min_conf: float) -> bool:
    if idx < 0 or idx >= coco3d.shape[0]:
        return False
    if float(coco3d[idx, 3]) < min_conf:
        return False
    xyz = coco3d[idx, :3]
    return bool(np.all(np.isfinite(xyz)) and np.linalg.norm(xyz) > 1e-4)


def _fabrik_chain(
    rest_local: np.ndarray,
    bone_lengths: np.ndarray,
    chain: list[int],
    root_target: np.ndarray,
    end_target: np.ndarray,
    *,
    iterations: int = 8,
) -> np.ndarray:
    """FABRIK 位置 IK，返回链上关节世界坐标。"""
    n = len(chain)
    if n < 2:
        return np.zeros((n, 3), dtype=np.float64)
    joints = np.zeros((n, 3), dtype=np.float64)
    joints[0] = root_target
    direction = normalize(end_target - root_target)
    if float(np.linalg.norm(direction)) < 1e-6:
        direction = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    dist = 0.0
    for i in range(1, n):
        dist += float(bone_lengths[chain[i]])
    dist = max(dist, 1e-4)
    for i in range(1, n):
        joints[i] = joints[i - 1] + direction * float(bone_lengths[chain[i]])

    lens = [float(bone_lengths[chain[i]]) for i in range(1, n)]
    for _ in range(iterations):
        joints[-1] = end_target
        for i in range(n - 2, -1, -1):
            d = normalize(joints[i] - joints[i + 1])
            joints[i] = joints[i + 1] + d * lens[i]
        joints[0] = root_target
        for i in range(1, n):
            d = normalize(joints[i] - joints[i - 1])
            joints[i] = joints[i - 1] + d * lens[i - 1]
    return joints


def _set_joint_rotation_from_direction(
    pose: np.ndarray,
    skeleton: SomaSkeleton,
    joint_idx: int,
    parent_world_rot: np.ndarray,
    direction_world: np.ndarray,
) -> np.ndarray:
    rest_dir = skeleton.rest_local[joint_idx].astype(np.float64)
    rest_dir_p = parent_world_rot @ rest_dir
    rot_world = rotation_between(rest_dir_p, direction_world)
    local_rot = parent_world_rot.T @ rot_world @ parent_world_rot
    pose[skeleton.pose_joint_index(joint_idx)] = axis_angle_from_matrix(local_rot)
    return parent_world_rot @ local_rot


def _set_joint_rotation_for_child_bone(
    pose: np.ndarray,
    skeleton: SomaSkeleton,
    joint_idx: int,
    child_idx: int,
    parent_world_rot: np.ndarray,
    direction_world: np.ndarray,
) -> np.ndarray:
    """Rotate joint_idx，使子骨（joint_idx→child_idx）指向 direction_world。"""
    target = normalize(np.asarray(direction_world, dtype=np.float64).reshape(3))
    if float(np.linalg.norm(target)) < 1e-8:
        return parent_world_rot
    rest_child = skeleton.rest_local[child_idx].astype(np.float64)
    local_target = parent_world_rot.T @ target
    local_rot = rotation_between(rest_child, local_target)
    pose[skeleton.pose_joint_index(joint_idx)] = axis_angle_from_matrix(local_rot)
    return parent_world_rot @ local_rot


def _swing_ik_chain_to_targets(
    pose: np.ndarray,
    skeleton: SomaSkeleton,
    chain: list[int],
    trans: np.ndarray,
    targets: list[np.ndarray],
) -> None:
    """逐节旋转 chain，使子关节朝向 COCO 目标点（FK 感知 child-bone IK）。"""
    trans_use = np.asarray(trans, dtype=np.float32).reshape(3)
    joints, world_rots = fk_forward(pose, trans_use, skeleton)
    for k in range(min(len(targets), len(chain) - 1)):
        j = chain[k]
        child = chain[k + 1]
        origin = joints[j].astype(np.float64)
        direction = targets[k].astype(np.float64) - origin
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


def swing_ik_limbs(
    skeleton: SomaSkeleton,
    coco3d: np.ndarray,
    pose: np.ndarray,
    trans: np.ndarray,
    *,
    min_conf: float = 0.25,
) -> None:
    """HybrIK-style swing：用 COCO 三角化端点初始化四肢 pose（保留已有脊柱 pose）。"""
    trans_use = np.asarray(trans, dtype=np.float64).reshape(3)

    for name, i_sh, i_el, i_wr in _LIMB_COCO:
        chain = list(skeleton.limb_chains.get(name, ()))
        if len(chain) < 3:
            continue
        if not (
            _valid_coco_point(coco3d, i_sh, min_conf)
            and _valid_coco_point(coco3d, i_el, min_conf)
            and _valid_coco_point(coco3d, i_wr, min_conf)
        ):
            continue
        p_s = coco3d[i_sh, :3].astype(np.float64)
        p_e = coco3d[i_el, :3].astype(np.float64)
        p_w = coco3d[i_wr, :3].astype(np.float64)

        if name.endswith("_leg"):
            # SOMA：LeftLeg(68) 短骨；膝在 LeftShin(69)，踝在 LeftFoot(70)
            j_hip = chain[0]
            j_knee = chain[1]
            j_ankle = chain[2]
            _swing_ik_chain_to_targets(
                pose,
                skeleton,
                [j_hip, j_knee, j_ankle],
                trans_use,
                [p_e, p_w],
            )
            continue

        # 肩→肘→腕：child-bone IK（12→13 对准 COCO 肩，非 rest[12] 锁骨）
        j_clav = chain[0]
        j_upper = chain[1]
        j_fore = chain[2]
        j_hand = chain[3] if len(chain) > 3 else chain[2]
        _swing_ik_chain_to_targets(
            pose,
            skeleton,
            [j_clav, j_upper, j_fore, j_hand],
            trans_use,
            [p_s, p_e, p_w],
        )


def fabrik_ik_spine(
    skeleton: SomaSkeleton,
    coco3d: np.ndarray,
    pose: np.ndarray,
    trans: np.ndarray,
    *,
    min_conf: float = 0.25,
    iterations: int = 8,
) -> None:
    chain = list(skeleton.spine_chain)
    if len(chain) < 2:
        return
    if _valid_coco_point(coco3d, COCO["l_hip"], min_conf) and _valid_coco_point(
        coco3d, COCO["r_hip"], min_conf
    ):
        root = 0.5 * (
            coco3d[COCO["l_hip"], :3].astype(np.float64)
            + coco3d[COCO["r_hip"], :3].astype(np.float64)
        )
    else:
        root = trans.astype(np.float64)
    if _valid_coco_point(coco3d, COCO["nose"], min_conf):
        end = coco3d[COCO["nose"], :3].astype(np.float64)
    elif _valid_coco_point(coco3d, COCO["l_shoulder"], min_conf) and _valid_coco_point(
        coco3d, COCO["r_shoulder"], min_conf
    ):
        end = 0.5 * (
            coco3d[COCO["l_shoulder"], :3].astype(np.float64)
            + coco3d[COCO["r_shoulder"], :3].astype(np.float64)
        )
    else:
        return

    positions = _fabrik_chain(
        skeleton.rest_local,
        skeleton.bone_lengths,
        chain,
        root,
        end,
        iterations=iterations,
    )
    _, world_rots = fk_forward(pose, trans, skeleton)
    for k in range(1, len(chain)):
        j = chain[k]
        p = int(skeleton.parent[j])
        direction = positions[k] - positions[k - 1]
        world_rots[j] = _set_joint_rotation_from_direction(
            pose, skeleton, j, world_rots[p], direction
        )


def analytical_ik_init(
    skeleton: SomaSkeleton,
    coco3d: np.ndarray,
    *,
    min_conf: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Step 1：解析 IK → pose (77,3), trans (3,)。"""
    pose = np.zeros((77, 3), dtype=np.float32)
    trans = np.zeros(3, dtype=np.float32)
    if _valid_coco_point(coco3d, COCO["l_hip"], min_conf) and _valid_coco_point(
        coco3d, COCO["r_hip"], min_conf
    ):
        trans = (
            0.5
            * (
                coco3d[COCO["l_hip"], :3].astype(np.float32)
                + coco3d[COCO["r_hip"], :3].astype(np.float32)
            )
        ).astype(np.float32)
    fabrik_ik_spine(skeleton, coco3d, pose, trans, min_conf=min_conf)
    swing_ik_limbs(skeleton, coco3d, pose, trans, min_conf=min_conf)
    return pose, trans
