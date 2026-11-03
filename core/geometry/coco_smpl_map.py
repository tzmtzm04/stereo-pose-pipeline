"""COCO-17 3D → SMPL-24 关节位置（社区映射，无 smplx 依赖）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from core.geometry.joint_maps import BodyJointMaps, default_body_joint_maps


def _valid(coco: np.ndarray, idx: int, min_conf: float) -> bool:
    if idx < 0 or idx >= coco.shape[0]:
        return False
    if float(coco[idx, 3]) < min_conf:
        return False
    xyz = coco[idx, :3]
    return bool(np.all(np.isfinite(xyz)))


def _copy_coco(
    out: np.ndarray,
    smpl_idx: int,
    coco: np.ndarray,
    coco_idx: int,
) -> None:
    out[smpl_idx, :3] = coco[coco_idx, :3]
    out[smpl_idx, 3] = coco[coco_idx, 3]


def _resolve_ref(
    spec: int | dict[str, Any],
    coco: np.ndarray,
    smpl_buf: np.ndarray,
    *,
    min_conf: float,
) -> tuple[np.ndarray, float] | None:
    if isinstance(spec, int):
        si = int(spec)
        if si < 0 or si >= smpl_buf.shape[0]:
            return None
        if float(smpl_buf[si, 3]) <= 0:
            return None
        return smpl_buf[si, :3].astype(np.float64), float(smpl_buf[si, 3])

    kind = str(spec.get("type", ""))
    if kind == "coco":
        ci = int(spec["index"])
        if not _valid(coco, ci, min_conf):
            return None
        return coco[ci, :3].astype(np.float64), float(coco[ci, 3])

    if kind in ("mid", "coco_mid"):
        idxs = [int(x) for x in spec["indices"]]
        pts, confs = [], []
        for ci in idxs:
            if not _valid(coco, ci, min_conf):
                continue
            pts.append(coco[ci, :3].astype(np.float64))
            confs.append(float(coco[ci, 3]))
        if not pts:
            return None
        return np.mean(pts, axis=0), float(np.min(confs))

    if kind == "smpl":
        si = int(spec["index"])
        if si < 0 or si >= smpl_buf.shape[0]:
            return None
        if float(smpl_buf[si, 3]) <= 0:
            return None
        return smpl_buf[si, :3].astype(np.float64), float(smpl_buf[si, 3])

    if kind == "lerp_points":
        a = _resolve_ref(spec["a"], coco, smpl_buf, min_conf=min_conf)
        b = _resolve_ref(spec["b"], coco, smpl_buf, min_conf=min_conf)
        if a is None or b is None:
            return None
        t = float(spec.get("t", 0.5))
        xyz = (1.0 - t) * a[0] + t * b[0]
        return xyz, float(min(a[1], b[1]))

    if kind == "lerp_chain":
        a = _resolve_ref(spec["from"], coco, smpl_buf, min_conf=min_conf)
        b = _resolve_ref(spec["to"], coco, smpl_buf, min_conf=min_conf)
        if a is None or b is None:
            return None
        t = float(spec.get("t", 0.5))
        xyz = (1.0 - t) * a[0] + t * b[0]
        return xyz, float(min(a[1], b[1]))

    if kind == "copy":
        return _resolve_ref(spec["from"], coco, smpl_buf, min_conf=min_conf)

    return None


def coco17_to_smpl24_joints(
    coco3d: np.ndarray,
    *,
    maps: BodyJointMaps | None = None,
    min_conf: float = 0.25,
) -> np.ndarray:
    """COCO (17,4) OpenCV 米制 → SMPL-24 (24,4) 关节位置 + conf。"""
    maps = maps or default_body_joint_maps()
    coco = np.asarray(coco3d, dtype=np.float32).reshape(-1, 4)
    out = np.zeros((24, 4), dtype=np.float32)

    for smpl_idx, coco_idx in maps.coco_to_smpl_direct.items():
        if not _valid(coco, int(coco_idx), min_conf):
            continue
        _copy_coco(out, int(smpl_idx), coco, int(coco_idx))

    for smpl_idx in (0, 12, 3, 6, 9, 13, 14, 10, 11):
        spec = maps.coco_to_smpl_derived.get(int(smpl_idx))
        if spec is None:
            continue
        resolved = _resolve_ref(spec, coco, out, min_conf=min_conf)
        if resolved is None:
            continue
        out[int(smpl_idx), :3] = resolved[0].astype(np.float32)
        out[int(smpl_idx), 3] = float(resolved[1])

    # hand 与 wrist 同位置
    for wrist, hand in ((20, 22), (21, 23)):
        if float(out[wrist, 3]) > 0 and float(out[hand, 3]) <= 0:
            out[hand] = out[wrist]

    return out


# SMPL-24 → COCO-17 直接观测关节（用于 FK 对比输出）
_SMPL_TO_COCO_DIRECT: tuple[tuple[int, int], ...] = (
    (15, 0),   # head → nose
    (16, 5),   # left_shoulder
    (17, 6),
    (18, 7),
    (19, 8),
    (20, 9),
    (21, 10),
    (1, 11),
    (2, 12),
    (4, 13),
    (5, 14),
    (7, 15),
    (8, 16),
)


def smpl24_to_coco17_joints(
    smpl24: np.ndarray,
    *,
    template_coco: np.ndarray | None = None,
    template_conf: np.ndarray | None = None,
) -> np.ndarray:
    """SMPL-24 → COCO-17；未覆盖关节保留 template（避免眼/耳置零）。"""
    smpl = np.asarray(smpl24, dtype=np.float32).reshape(24, 4)
    if template_coco is not None:
        out = np.asarray(template_coco, dtype=np.float32).reshape(17, 4).copy()
    else:
        out = np.zeros((17, 4), dtype=np.float32)
        if template_conf is not None:
            out[:, 3] = np.asarray(template_conf, dtype=np.float32).reshape(17)
    for si, ci in _SMPL_TO_COCO_DIRECT:
        if float(smpl[si, 3]) <= 0:
            continue
        out[ci, :3] = smpl[si, :3]
        out[ci, 3] = max(float(out[ci, 3]), float(smpl[si, 3]))
    return out


def coco17_from_soma_fk_or_smpl(
    joints78: np.ndarray,
    smpl24: np.ndarray,
    skeleton,
    *,
    maps: BodyJointMaps | None = None,
    template_conf: np.ndarray | None = None,
) -> np.ndarray:
    """SOMA FK 关节经 SMPL 表映回 COCO-17（对比用）。"""
    from core.geometry.body_fk import coco_joints_from_fk

    maps = maps or default_body_joint_maps()
    smpl = np.asarray(smpl24, dtype=np.float32).reshape(24, 4)
    j = np.asarray(joints78, dtype=np.float32).reshape(-1, 3)
    out = np.zeros((17, 4), dtype=np.float32)
    if template_conf is not None:
        out[:, 3] = np.asarray(template_conf, dtype=np.float32).reshape(17)
    for si, ci in _SMPL_TO_COCO_DIRECT:
        sj = int(maps.smpl24_to_soma_joint[si])
        off = maps.smpl24_to_soma_offset[si]
        if sj >= j.shape[0]:
            continue
        conf = float(smpl[si, 3])
        if conf <= 0:
            continue
        out[ci, :3] = j[sj] + off.astype(np.float32)
        out[ci, 3] = max(float(out[ci, 3]), conf)
    fk = coco_joints_from_fk(joints78, skeleton)
    for i in range(17):
        if float(out[i, 3]) <= 0:
            out[i, :3] = fk[i, :3]
            out[i, 3] = 1.0
    return out


def smpl24_joints_to_soma_targets(
    smpl24: np.ndarray,
    *,
    maps: BodyJointMaps | None = None,
) -> dict[int, np.ndarray]:
    """SMPL-24 关节 → SOMA 关节索引 → 世界目标 (3,)。"""
    maps = maps or default_body_joint_maps()
    smpl = np.asarray(smpl24, dtype=np.float32).reshape(24, 4)
    targets: dict[int, np.ndarray] = {}
    for si in range(24):
        if float(smpl[si, 3]) <= 0:
            continue
        sj = int(maps.smpl24_to_soma_joint[si])
        off = maps.smpl24_to_soma_offset[si]
        xyz = smpl[si, :3].astype(np.float64) + off.astype(np.float64)
        prev = targets.get(sj)
        conf = float(smpl[si, 3])
        if prev is None or conf >= 0.5:
            targets[sj] = xyz
    return targets
