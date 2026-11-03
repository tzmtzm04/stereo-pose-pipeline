"""从 WB133 3D 点推导 mink SE3 朝向目标（脚板、骨轴）。"""

from __future__ import annotations

import numpy as np

# 左相机 OpenCV：Y 向下，站立时「上」为 -Y
_WORLD_UP = np.array([0.0, -1.0, 0.0], dtype=np.float64)


def _normalize(v: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray | None:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None if fallback is None else fallback.copy()
    return (v / n).astype(np.float64)


def rotation_from_forward_up(forward: np.ndarray, up_hint: np.ndarray) -> np.ndarray | None:
    """列向量 [forward, lateral, up] → 3×3。"""
    f = _normalize(np.asarray(forward, dtype=np.float64))
    if f is None:
        return None
    u_hint = _normalize(np.asarray(up_hint, dtype=np.float64), _WORLD_UP)
    assert u_hint is not None
    lat = _normalize(np.cross(u_hint, f))
    if lat is None:
        lat = _normalize(np.cross(f, np.array([1.0, 0.0, 0.0], dtype=np.float64)))
    if lat is None:
        return None
    up = np.cross(f, lat)
    up = _normalize(up, u_hint)
    assert up is not None
    return np.column_stack([f, lat, up])


def foot_heel_se3(
    heel: np.ndarray,
    big_toe: np.ndarray,
    small_toe: np.ndarray,
) -> object | None:
    """脚跟 site：位置=heel，朝向=趾根方向 + 脚面 up。"""
    try:
        from mink import SE3
    except ImportError:
        return None
    heel = np.asarray(heel, dtype=np.float64).reshape(3)
    toe_mid = 0.5 * (
        np.asarray(big_toe, dtype=np.float64).reshape(3)
        + np.asarray(small_toe, dtype=np.float64).reshape(3)
    )
    forward = toe_mid - heel
    r = rotation_from_forward_up(forward, _WORLD_UP)
    if r is None:
        return SE3.from_translation(heel)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = r
    mat[:3, 3] = heel
    return SE3.from_matrix(mat)


def bone_direction_se3(root: np.ndarray, tip: np.ndarray) -> object | None:
    """骨段 tip site：位置=tip，x 轴沿 root→tip。"""
    try:
        from mink import SE3
    except ImportError:
        return None
    root = np.asarray(root, dtype=np.float64).reshape(3)
    tip = np.asarray(tip, dtype=np.float64).reshape(3)
    forward = tip - root
    r = rotation_from_forward_up(forward, _WORLD_UP)
    if r is None:
        return SE3.from_translation(tip)
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = r
    mat[:3, 3] = tip
    return SE3.from_matrix(mat)


def site_frame_target(
    site_name: str,
    active: dict[str, np.ndarray],
) -> object | None:
    """按 site 名返回 SE3 目标；无朝向信息则仅平移。"""
    try:
        from mink import SE3
    except ImportError:
        return None
    pos = active.get(site_name)
    if pos is None:
        return None

    if site_name == "left_heel":
        if "left_big_toe" in active and "left_small_toe" in active:
            return foot_heel_se3(pos, active["left_big_toe"], active["left_small_toe"])
    if site_name == "right_heel":
        if "right_big_toe" in active and "right_small_toe" in active:
            return foot_heel_se3(pos, active["right_big_toe"], active["right_small_toe"])

    limb_pairs = (
        ("left_wrist", "left_elbow"),
        ("right_wrist", "right_elbow"),
        ("left_ankle", "left_knee"),
        ("right_ankle", "right_knee"),
        ("left_elbow", "left_shoulder"),
        ("right_elbow", "right_shoulder"),
        ("left_knee", "left_hip"),
        ("right_knee", "right_hip"),
    )
    for tip, root_name in limb_pairs:
        if site_name == tip and root_name in active:
            return bone_direction_se3(active[root_name], pos)

    return SE3.from_translation(np.asarray(pos, dtype=np.float64))


def com_target_from_active(active: dict[str, np.ndarray]) -> np.ndarray | None:
    if "pelvis" in active:
        return np.asarray(active["pelvis"], dtype=np.float64).reshape(3).copy()
    if "left_hip" in active and "right_hip" in active:
        return 0.5 * (
            np.asarray(active["left_hip"], dtype=np.float64)
            + np.asarray(active["right_hip"], dtype=np.float64)
        )
    return None


def orientation_cost_for_site(site_name: str, pos_w: float, ori_w: float) -> np.ndarray:
    """FrameTask orientation_cost 三维权重。"""
    foot = {"left_heel", "right_heel", "left_big_toe", "right_big_toe"}
    limb_tip = {
        "left_wrist",
        "right_wrist",
        "left_ankle",
        "right_ankle",
        "left_elbow",
        "right_elbow",
        "left_knee",
        "right_knee",
    }
    if site_name in foot:
        return np.full(3, ori_w * 1.2, dtype=np.float64)
    if site_name in limb_tip:
        return np.array([ori_w, ori_w * 0.35, ori_w * 0.35], dtype=np.float64)
    return np.zeros(3, dtype=np.float64)
