"""COCO → SMPL-24 → SOMA 映射管线（社区路子）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.geometry.body_fk import coco_joints_from_fk, fk_forward_aligned
from core.geometry.body_ik import analytical_ik_init
from core.geometry.coco_smpl_map import (
    coco17_to_smpl24_joints,
    smpl24_to_coco17_joints,
)
from core.geometry.joint_maps import BodyJointMaps, default_body_joint_maps
from core.geometry.smpl_soma_ik import smpl24_usable_for_ik, smpl_targets_mean_error
from core.geometry.soma_skeleton import SomaSkeleton, default_soma_skeleton
from core.geometry.stereo_body_pipeline import tri_usable_for_body_ik


@dataclass
class SmplSomaOutput:
    kpts3d: np.ndarray  # (17, 4) COCO FK
    smpl24: np.ndarray  # (24, 4)
    joints78: np.ndarray
    pose77: np.ndarray
    trans: np.ndarray
    smpl_map_error_m: float


class SmplSomaPipeline:
    """tri COCO-17 → SMPL 关节 → SOMA IK + FK。"""

    def __init__(
        self,
        skeleton: SomaSkeleton | None = None,
        maps: BodyJointMaps | None = None,
        *,
        min_conf: float = 0.25,
    ) -> None:
        self._skeleton = skeleton or default_soma_skeleton()
        self._maps = maps or default_body_joint_maps()
        self._min_conf = float(min_conf)

    def process_coco17(
        self,
        coco3d: np.ndarray,
    ) -> SmplSomaOutput | None:
        coco = np.asarray(coco3d, dtype=np.float32).reshape(-1, 4)
        if coco.shape[0] < 17:
            return None
        if not tri_usable_for_body_ik(coco, min_conf=self._min_conf):
            return None

        smpl24 = coco17_to_smpl24_joints(
            coco,
            maps=self._maps,
            min_conf=self._min_conf,
        )
        if not smpl24_usable_for_ik(smpl24, min_conf=self._min_conf):
            return None

        pseudo_coco = smpl24_to_coco17_joints(
            smpl24,
            template_coco=coco[:17],
        )
        pose, trans = analytical_ik_init(
            self._skeleton,
            pseudo_coco,
            min_conf=self._min_conf,
        )
        joints, _, trans = fk_forward_aligned(
            pose, trans, coco, self._skeleton
        )
        kpts_out = np.zeros((17, 4), dtype=np.float32)
        kpts_out[:, :3] = coco_joints_from_fk(joints, self._skeleton)[:17]
        kpts_out[:, 3] = coco[:17, 3]

        err = smpl_targets_mean_error(joints, smpl24, maps=self._maps)
        return SmplSomaOutput(
            kpts3d=kpts_out,
            smpl24=smpl24,
            joints78=joints,
            pose77=pose,
            trans=trans.astype(np.float32),
            smpl_map_error_m=err,
        )
