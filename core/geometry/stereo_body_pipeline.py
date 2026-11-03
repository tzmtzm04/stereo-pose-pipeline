"""双目 SOMA 约束管线（文档 v1.3 Python MVP）。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.geometry.body_fk import coco_joints_from_fk, fk_forward, fk_forward_aligned
from core.geometry.body_ik import analytical_ik_init
from core.geometry.body_optimize import (
    BodyLossWeights,
    BodyOptimizerConfig,
    optimize_pose,
)
from core.geometry.calib_model import StereoCalibration
from core.geometry.soma_skeleton import SomaSkeleton, default_soma_skeleton


# COCO 索引：body-ik 门控用
_COCO_L_HIP = 11
_COCO_R_HIP = 12


def tri_usable_for_body_ik(
    kpts3d: np.ndarray,
    *,
    min_conf: float = 0.25,
    min_valid: int = 6,
    min_depth_m: float = 0.5,
    max_depth_m: float = 8.0,
) -> bool:
    """tri 有效深度点不足或缺髋时不跑 IK，回退 raw tri。"""
    pts = np.asarray(kpts3d, dtype=np.float32).reshape(-1, 4)
    if pts.shape[0] < 17:
        return False
    valid = (
        (pts[:, 3] >= float(min_conf))
        & np.all(np.isfinite(pts[:, :3]), axis=1)
        & (pts[:, 2] > float(min_depth_m))
        & (pts[:, 2] < float(max_depth_m))
    )
    if int(valid.sum()) < int(min_valid):
        return False
    return bool(valid[_COCO_L_HIP] or valid[_COCO_R_HIP])


@dataclass
class StereoBodyConfig:
    min_conf: float = 0.25
    weights: BodyLossWeights = field(default_factory=BodyLossWeights)
    optimizer: BodyOptimizerConfig = field(default_factory=BodyOptimizerConfig)


@dataclass
class StereoBodyOutput:
    kpts3d: np.ndarray  # (17, 4) 矫正 COCO 3D
    joints78: np.ndarray  # (78, 3)
    pose77: np.ndarray  # (77, 3)
    trans: np.ndarray  # (3,)
    final_loss: float
    tri_ok: bool


class StereoBodyPipeline:
    """Step 0（外部 tri）→ Step 1 IK → Step 2 Adam → 输出矫正 COCO 3D。"""

    def __init__(
        self,
        calib: StereoCalibration,
        skeleton: SomaSkeleton | None = None,
        cfg: StereoBodyConfig | None = None,
    ) -> None:
        self._calib = calib
        self._skeleton = skeleton or default_soma_skeleton()
        self._cfg = cfg or StereoBodyConfig()
        self._frame_id = 0
        self._pose_prev: np.ndarray | None = None
        self._pose_prev2: np.ndarray | None = None

    def reset(self) -> None:
        self._frame_id = 0
        self._pose_prev = None
        self._pose_prev2 = None

    def _pack_coco_output(
        self,
        joints: np.ndarray,
        pose: np.ndarray,
        trans: np.ndarray,
        kl: np.ndarray,
        kr: np.ndarray,
        *,
        final_loss: float,
    ) -> StereoBodyOutput:
        coco_corr = coco_joints_from_fk(joints, self._skeleton)
        conf = np.minimum(kl[:17, 2], kr[:17, 2])
        kpts_out = np.zeros((17, 4), dtype=np.float32)
        kpts_out[:, :3] = coco_corr[:17]
        kpts_out[:, 3] = conf
        return StereoBodyOutput(
            kpts3d=kpts_out,
            joints78=joints,
            pose77=pose,
            trans=trans.astype(np.float32),
            final_loss=float(final_loss),
            tri_ok=True,
        )

    def process_frame_ik_only(
        self,
        det_left: dict | None,
        det_right: dict | None,
        kpts3d_tri: np.ndarray | None,
    ) -> StereoBodyOutput | None:
        """轻档：解析 IK + FK，无 Adam（实时 infer 用）。"""
        if (
            det_left is None
            or det_right is None
            or kpts3d_tri is None
            or float(np.asarray(kpts3d_tri)[:, 3].sum()) <= 0
        ):
            return None

        kl = np.asarray(det_left["keypoints"], dtype=np.float32)
        kr = np.asarray(det_right["keypoints"], dtype=np.float32)
        coco3d = np.asarray(kpts3d_tri, dtype=np.float32).reshape(-1, 4)
        if coco3d.shape[0] < 17:
            return None

        pose, trans = analytical_ik_init(
            self._skeleton,
            coco3d,
            min_conf=self._cfg.min_conf,
        )
        joints, _, trans = fk_forward_aligned(
            pose, trans, coco3d, self._skeleton
        )
        return self._pack_coco_output(
            joints, pose, trans, kl, kr, final_loss=0.0
        )

    def process_frame(
        self,
        det_left: dict | None,
        det_right: dict | None,
        kpts3d_tri: np.ndarray | None,
    ) -> StereoBodyOutput | None:
        if (
            det_left is None
            or det_right is None
            or kpts3d_tri is None
            or float(np.asarray(kpts3d_tri)[:, 3].sum()) <= 0
        ):
            return None

        kl = np.asarray(det_left["keypoints"], dtype=np.float32)
        kr = np.asarray(det_right["keypoints"], dtype=np.float32)
        coco3d = np.asarray(kpts3d_tri, dtype=np.float32).reshape(-1, 4)
        if coco3d.shape[0] < 17:
            return None

        pose_init, trans_init = analytical_ik_init(
            self._skeleton,
            coco3d,
            min_conf=self._cfg.min_conf,
        )
        pose_opt, trans_opt, loss = optimize_pose(
            pose_init,
            trans_init,
            self._pose_prev,
            self._pose_prev2,
            self._frame_id,
            kl[:, :2],
            kr[:, :2],
            kl[:, 2],
            kr[:, 2],
            self._skeleton,
            self._calib,
            weights=self._cfg.weights,
            opt_cfg=self._cfg.optimizer,
            min_conf=self._cfg.min_conf,
        )
        joints, _, trans_opt = fk_forward_aligned(
            pose_opt, trans_opt, coco3d, self._skeleton
        )

        self._pose_prev2 = self._pose_prev.copy() if self._pose_prev is not None else None
        self._pose_prev = pose_opt.copy()
        self._frame_id += 1

        return self._pack_coco_output(
            joints, pose_opt, trans_opt, kl, kr, final_loss=loss
        )
