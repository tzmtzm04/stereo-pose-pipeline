"""腕/踝：观测方向 + 模型骨长定位，子链反解末端关节（不进主 IK task）。

前臂用 wrist 球关节；小腿踝 site 在 shin 上，由 knee 铰链定向（ankle 球关节只控脚掌）。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.offline.two_stage.coco_wholebody_layout import NUM_KEYPOINTS_BODY_FEET


@dataclass(frozen=True)
class ExtremitySpec:
    anchor_site: str
    tip_site: str
    joint_name: str
    wb_anchor_idx: int
    wb_tip_idx: int


@dataclass(frozen=True)
class ExtremityIKConfig:
    enabled: bool = False
    min_conf: float = 0.35
    max_iter: int = 50
    damping: float = 0.005
    step_scale: float = 0.95
    dq_clip: float = 0.25
    tol: float = 2e-3


EXTREMITY_SPECS: tuple[ExtremitySpec, ...] = (
    ExtremitySpec("left_elbow", "left_wrist", "left_wrist", 7, 9),
    ExtremitySpec("right_elbow", "right_wrist", "right_wrist", 8, 10),
    ExtremitySpec("left_knee", "left_ankle", "left_knee", 13, 15),
    ExtremitySpec("right_knee", "right_ankle", "right_knee", 14, 16),
)

# 踝球关节：脚掌 site 沿 obs 踝→趾方向（不参与主 task）
FOOT_SPECS: tuple[ExtremitySpec, ...] = (
    ExtremitySpec("left_ankle", "left_big_toe", "left_ankle", 15, 17),
    ExtremitySpec("right_ankle", "right_big_toe", "right_ankle", 16, 20),
)


def _wb_point(
    kpts3d: np.ndarray,
    idx: int,
    *,
    min_conf: float,
) -> np.ndarray | None:
    k = np.asarray(kpts3d, dtype=np.float64).reshape(-1, 4)
    if idx < 0 or idx >= k.shape[0]:
        return None
    if float(k[idx, 3]) < min_conf:
        return None
    xyz = k[idx, :3]
    if not np.all(np.isfinite(xyz)) or np.linalg.norm(xyz) <= 1e-4:
        return None
    return xyz.copy()


def _unit_vector(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray | None:
    n = float(np.linalg.norm(vec))
    if n >= 1e-6:
        return vec / n
    fb = np.asarray(fallback, dtype=np.float64).reshape(3)
    fn = float(np.linalg.norm(fb))
    if fn >= 1e-6:
        return fb / fn
    return None


def scaled_extremity_target(
    anchor_ik: np.ndarray,
    obs_tip: np.ndarray,
    obs_anchor: np.ndarray | None,
    segment_len_m: float,
    *,
    dir_fallback: np.ndarray,
) -> np.ndarray | None:
    """锚点 = IK FK；方向 = obs 骨段连线；长度 = 模型当前骨段长。"""
    if obs_anchor is not None:
        direction_src = obs_tip - obs_anchor
    else:
        direction_src = obs_tip - anchor_ik
    direction = _unit_vector(direction_src, dir_fallback)
    if direction is None:
        return None
    return np.asarray(anchor_ik, dtype=np.float64) + float(segment_len_m) * direction


class ExtremityIKSolver:
    """主 IK 之后：解析目标位置 + 只动 wrist/ankle 关节的子链 DLS。"""

    def __init__(
        self,
        model_path: Path | str,
        cfg: ExtremityIKConfig | None = None,
    ) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self.cfg = cfg or ExtremityIKConfig()

        self._site_ids: dict[str, int] = {}
        self._joint_v_ids: dict[str, list[int]] = {}
        all_specs = EXTREMITY_SPECS + FOOT_SPECS
        for spec in all_specs:
            for site_name in (spec.anchor_site, spec.tip_site):
                if site_name in self._site_ids:
                    continue
                sid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_SITE, site_name
                )
                if sid < 0:
                    raise ValueError(f"MJCF 缺少 site: {site_name}")
                self._site_ids[site_name] = int(sid)

            jid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, spec.joint_name
            )
            if jid < 0:
                raise ValueError(f"MJCF 缺少 joint: {spec.joint_name}")
            v0 = int(self.model.jnt_dofadr[jid])
            jtype = int(self.model.jnt_type[jid])
            if jtype == mujoco.mjtJoint.mjJNT_BALL:
                vdim = 3
            elif jtype == mujoco.mjtJoint.mjJNT_HINGE:
                vdim = 1
            else:
                vdim = max(1, int(self.model.nv - v0))
            self._joint_v_ids[spec.joint_name] = list(range(v0, v0 + vdim))

    def apply(
        self,
        qpos: np.ndarray,
        kpts3d: np.ndarray,
        *,
        min_conf: float | None = None,
    ) -> np.ndarray:
        cfg = self.cfg
        mc = float(cfg.min_conf if min_conf is None else min_conf)
        if not cfg.enabled:
            return np.asarray(qpos, dtype=np.float64).copy()

        k = np.asarray(kpts3d, dtype=np.float64)
        if k.reshape(-1, 4).shape[0] < NUM_KEYPOINTS_BODY_FEET:
            return np.asarray(qpos, dtype=np.float64).copy()

        out = np.asarray(qpos, dtype=np.float64).copy()
        self.data.qpos[:] = out
        self._mujoco.mj_forward(self.model, self.data)

        for spec_group in (EXTREMITY_SPECS, FOOT_SPECS):
            for spec in spec_group:
                obs_tip = _wb_point(k, spec.wb_tip_idx, min_conf=mc)
                if obs_tip is None:
                    continue
                obs_anchor = _wb_point(k, spec.wb_anchor_idx, min_conf=mc)

                anchor_sid = self._site_ids[spec.anchor_site]
                tip_sid = self._site_ids[spec.tip_site]
                anchor = self.data.site_xpos[anchor_sid].copy()
                tip_cur = self.data.site_xpos[tip_sid].copy()
                seg_len = float(np.linalg.norm(tip_cur - anchor))
                if seg_len < 1e-6:
                    continue

                dir_fallback = tip_cur - anchor
                target = scaled_extremity_target(
                    anchor,
                    obs_tip,
                    obs_anchor,
                    seg_len,
                    dir_fallback=dir_fallback,
                )
                if target is None:
                    continue

                out = self._solve_tip_joint(
                    out,
                    tip_site=spec.tip_site,
                    joint_name=spec.joint_name,
                    target=target,
                )
                self.data.qpos[:] = out
                self._mujoco.mj_forward(self.model, self.data)

        return out

    def _solve_tip_joint(
        self,
        qpos: np.ndarray,
        *,
        tip_site: str,
        joint_name: str,
        target: np.ndarray,
    ) -> np.ndarray:
        cfg = self.cfg
        mujoco = self._mujoco
        self.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

        sid = self._site_ids[tip_site]
        v_ids = self._joint_v_ids[joint_name]
        tgt = np.asarray(target, dtype=np.float64).reshape(3)

        for _ in range(int(cfg.max_iter)):
            cur = self.data.site_xpos[sid]
            err = tgt - cur
            err_norm = float(np.linalg.norm(err))
            if err_norm < float(cfg.tol):
                break

            jacp = np.zeros((3, self.model.nv), dtype=np.float64)
            jacr = np.zeros((3, self.model.nv), dtype=np.float64)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, sid)
            j = jacp[:, v_ids]
            damp = float(cfg.damping)
            a = j @ j.T + (damp * damp) * np.eye(3, dtype=np.float64)
            dq_sub = j.T @ np.linalg.solve(a, err)
            dq_norm = float(np.linalg.norm(dq_sub))
            clip = float(cfg.dq_clip)
            if dq_norm > clip:
                dq_sub *= clip / dq_norm

            dq = np.zeros(self.model.nv, dtype=np.float64)
            dq[v_ids] = dq_sub
            mujoco.mj_integratePos(
                self.model,
                self.data.qpos,
                dq,
                float(cfg.step_scale),
            )
            mujoco.mj_forward(self.model, self.data)

        return self.data.qpos.copy()
