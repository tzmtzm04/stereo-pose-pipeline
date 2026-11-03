"""SOMA 78 关节 3D 预览布局（parent 树 → edges）。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.geometry.soma_skeleton import SomaSkeleton, default_soma_skeleton

# 关节分区（与 data/soma_skeleton.json 索引一致）
_SOMA_FACE_END = 12
_SOMA_LEFT_HAND_START = 12
_SOMA_RIGHT_HAND_START = 40
_SOMA_LEFT_LEG_START = 68
_SOMA_RIGHT_LEG_START = 73


@dataclass(frozen=True)
class SomaSkeleton3DRenderSpec:
    layout: str
    edges: list[tuple[int, int]]
    num_joints: int
    label: str
    joint_names: tuple[str, ...]
    center_indices: tuple[int, ...]


def soma_edges_from_parent(parent: np.ndarray) -> list[tuple[int, int]]:
    """parent[i] → (parent[i], i) 骨段。"""
    p = np.asarray(parent, dtype=np.int32).reshape(-1)
    edges: list[tuple[int, int]] = []
    for j in range(1, p.shape[0]):
        pj = int(p[j])
        if pj >= 0:
            edges.append((pj, j))
    return edges


def resolve_soma_skeleton3d_render_spec(
    skeleton: SomaSkeleton | None = None,
) -> SomaSkeleton3DRenderSpec:
    sk = skeleton or default_soma_skeleton()
    return SomaSkeleton3DRenderSpec(
        layout="soma78",
        edges=soma_edges_from_parent(sk.parent),
        num_joints=int(sk.num_joints),
        label="SOMA-78",
        joint_names=sk.joint_names,
        center_indices=(1, 12, 40, 68, 73),
    )


def joints78_to_kpts4(joints78: np.ndarray) -> np.ndarray:
    """(78,3) → (78,4)，conf 全 1。"""
    j = np.asarray(joints78, dtype=np.float32).reshape(-1, 3)
    if j.shape[0] < 78:
        out = np.zeros((78, 4), dtype=np.float32)
        n = j.shape[0]
        out[:n, :3] = j[:n]
        out[:n, 3] = 1.0
        return out
    out = np.zeros((j.shape[0], 4), dtype=np.float32)
    out[:, :3] = j[:, :3]
    out[:, 3] = 1.0
    return out


def resolve_joints78_from_record(
    rec: dict,
    skeleton: SomaSkeleton | None = None,
) -> np.ndarray | None:
    """jsonl 行 → (78,3)；优先 soma_joints78，否则 FK(pose77, trans)。"""
    raw = rec.get("soma_joints78")
    if raw is not None:
        j = np.asarray(raw, dtype=np.float32).reshape(-1, 3)
        if j.shape[0] >= 78 and np.any(np.isfinite(j)):
            return j[:78].copy()

    pose = rec.get("soma_pose77")
    trans = rec.get("soma_trans")
    if pose is None or trans is None:
        return None
    from core.geometry.body_fk import fk_forward

    sk = skeleton or default_soma_skeleton()
    joints, _ = fk_forward(
        np.asarray(pose, dtype=np.float64),
        np.asarray(trans, dtype=np.float64),
        sk,
    )
    return joints.astype(np.float32)


def joint_style_soma78(joint_idx: int, *, dimmed: bool = False) -> tuple[int, tuple[int, int, int]]:
    """分区着色：躯干 / 左臂 / 右臂 / 左腿 / 右腿 / 头面。"""
    if joint_idx == 0:
        return 4, (120, 120, 140) if dimmed else (160, 160, 180)
    if joint_idx < _SOMA_FACE_END:
        return 4, (150, 170, 190) if dimmed else (200, 220, 240)
    if _SOMA_LEFT_HAND_START <= joint_idx < _SOMA_RIGHT_HAND_START:
        return 3, (90, 150, 220) if dimmed else (120, 200, 255)
    if _SOMA_RIGHT_HAND_START <= joint_idx < _SOMA_LEFT_LEG_START:
        return 3, (90, 220, 150) if dimmed else (120, 255, 200)
    if _SOMA_LEFT_LEG_START <= joint_idx < _SOMA_RIGHT_LEG_START:
        return 4, (80, 180, 255) if dimmed else (100, 220, 255)
    if joint_idx >= _SOMA_RIGHT_LEG_START:
        return 4, (80, 255, 180) if dimmed else (100, 255, 220)
    return 5, (0, 150, 200) if dimmed else (0, 200, 255)
