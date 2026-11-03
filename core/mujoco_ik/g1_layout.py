"""COCO-17 / body → Unitree G1 mink IK 观测映射。"""

from __future__ import annotations

import numpy as np

from core.offline.two_stage.coco_wholebody_layout import NUM_KEYPOINTS_BODY

G1_HAND_SITES: tuple[str, str] = ("left_palm", "right_palm")
G1_FOOT_SITES: tuple[str, str] = ("left_foot", "right_foot")
G1_ELBOW_BODIES: tuple[str, str] = ("left_elbow_link", "right_elbow_link")
G1_KNEE_BODIES: tuple[str, str] = ("left_knee_link", "right_knee_link")
G1_SHOULDER_BODIES: tuple[str, str] = (
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
)
G1_HIP_BODIES: tuple[str, str] = ("left_hip_yaw_link", "right_hip_yaw_link")
G1_EXTREMITY_SITES: tuple[str, ...] = G1_HAND_SITES + G1_FOOT_SITES
G1_LIMB8_TASK_KEYS: tuple[str, ...] = (
    G1_HAND_SITES + G1_FOOT_SITES + G1_ELBOW_BODIES + G1_KNEE_BODIES
)
G1_OFFICIAL_LIMB_TASK_KEYS: tuple[str, ...] = G1_EXTREMITY_SITES
G1_FULL_LIMB_TASK_KEYS: tuple[str, ...] = (
    G1_HAND_SITES
    + G1_FOOT_SITES
    + G1_ELBOW_BODIES
    + G1_KNEE_BODIES
    + G1_SHOULDER_BODIES
    + G1_HIP_BODIES
)
# 兼容旧名
G1_LIMB_TASK_KEYS = G1_LIMB8_TASK_KEYS

G1_LIMB_FRAME_SPECS: dict[str, tuple[str, str]] = {
    "left_palm": ("left_palm", "site"),
    "right_palm": ("right_palm", "site"),
    "left_foot": ("left_foot", "site"),
    "right_foot": ("right_foot", "site"),
    "left_elbow_link": ("left_elbow_link", "body"),
    "right_elbow_link": ("right_elbow_link", "body"),
    "left_knee_link": ("left_knee_link", "body"),
    "right_knee_link": ("right_knee_link", "body"),
    "left_shoulder_yaw_link": ("left_shoulder_yaw_link", "body"),
    "right_shoulder_yaw_link": ("right_shoulder_yaw_link", "body"),
    "left_hip_yaw_link": ("left_hip_yaw_link", "body"),
    "right_hip_yaw_link": ("right_hip_yaw_link", "body"),
}

_COCO_WRIST = (9, 10)
_COCO_ANKLE = (15, 16)
_COCO_ELBOW = (7, 8)
_COCO_KNEE = (13, 14)
_COCO_SHOULDER = (5, 6)
_COCO_HIP = (11, 12)


def g1_limb_task_keys(
    *,
    official_only: bool = False,
    full_body: bool = False,
) -> tuple[str, ...]:
    if official_only:
        return G1_OFFICIAL_LIMB_TASK_KEYS
    if full_body:
        return G1_FULL_LIMB_TASK_KEYS
    return G1_LIMB8_TASK_KEYS


def _point_valid(kpts: np.ndarray, idx: int, min_conf: float) -> bool:
    if idx < 0 or idx >= kpts.shape[0]:
        return False
    if float(kpts[idx, 3]) < min_conf:
        return False
    xyz = kpts[idx, :3]
    return bool(np.all(np.isfinite(xyz)) and np.linalg.norm(xyz) > 1e-4)


def _get_point(kpts: np.ndarray, idx: int, min_conf: float) -> np.ndarray | None:
    if not _point_valid(kpts, idx, min_conf):
        return None
    return kpts[idx, :3].copy()


def coco_to_g1_targets(
    kpts3d: np.ndarray,
    *,
    min_conf: float = 0.35,
) -> dict[str, np.ndarray | None]:
    """(17+,4) → G1 FrameTask 目标 + 肩髋 meta（torso / COM）。"""
    k = np.asarray(kpts3d, dtype=np.float64).reshape(-1, 4)
    meta_keys = (
        *G1_FULL_LIMB_TASK_KEYS,
        "pelvis",
        "left_shoulder",
        "right_shoulder",
        "com",
    )
    if k.shape[0] < NUM_KEYPOINTS_BODY:
        return {name: None for name in meta_keys}

    ls = _get_point(k, _COCO_SHOULDER[0], min_conf)
    rs = _get_point(k, _COCO_SHOULDER[1], min_conf)
    lh = _get_point(k, _COCO_HIP[0], min_conf)
    rh = _get_point(k, _COCO_HIP[1], min_conf)

    pelvis: np.ndarray | None = None
    if lh is not None and rh is not None:
        pelvis = 0.5 * (lh + rh)
    elif lh is not None:
        pelvis = lh
    elif rh is not None:
        pelvis = rh

    return {
        "left_palm": _get_point(k, _COCO_WRIST[0], min_conf),
        "right_palm": _get_point(k, _COCO_WRIST[1], min_conf),
        "left_foot": _get_point(k, _COCO_ANKLE[0], min_conf),
        "right_foot": _get_point(k, _COCO_ANKLE[1], min_conf),
        "left_elbow_link": _get_point(k, _COCO_ELBOW[0], min_conf),
        "right_elbow_link": _get_point(k, _COCO_ELBOW[1], min_conf),
        "left_knee_link": _get_point(k, _COCO_KNEE[0], min_conf),
        "right_knee_link": _get_point(k, _COCO_KNEE[1], min_conf),
        "left_shoulder_yaw_link": ls,
        "right_shoulder_yaw_link": rs,
        "left_hip_yaw_link": lh,
        "right_hip_yaw_link": rh,
        "pelvis": pelvis,
        "left_shoulder": ls,
        "right_shoulder": rs,
        "com": pelvis,
    }


def describe_g1_layout(
    *,
    official_only: bool = False,
    full_body: bool = False,
) -> str:
    if official_only:
        return (
            "G1 mink 官方：palm/foot←腕踝 FrameTask；"
            "pelvis/torso orientation + Posture + COM"
        )
    if full_body:
        return (
            "G1 mink 全身 12 点：肩髋肘膝腕踝 FrameTask + "
            "pelvis/torso orientation + Posture + COM"
        )
    return (
        "G1 mink 8 点：palm/foot/elbow/knee FrameTask + "
        "pelvis/torso orientation + Posture + COM"
    )
