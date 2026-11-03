"""G1 qpos → 全身 stick figure FK（body + palm/foot site）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 与 COCO body 拓扑对齐：pelvis/torso/肩肘腕/髋膝踝
G1_BODY_FRAMES: tuple[tuple[str, str], ...] = (
    ("pelvis", "body"),
    ("torso_link", "body"),
    ("left_shoulder_yaw_link", "body"),
    ("left_elbow_link", "body"),
    ("left_palm", "site"),
    ("right_shoulder_yaw_link", "body"),
    ("right_elbow_link", "body"),
    ("right_palm", "site"),
    ("left_hip_yaw_link", "body"),
    ("left_knee_link", "body"),
    ("left_foot", "site"),
    ("right_hip_yaw_link", "body"),
    ("right_knee_link", "body"),
    ("right_foot", "site"),
)

G1_SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (0, 8),
    (0, 11),
    (8, 9),
    (9, 10),
    (11, 12),
    (12, 13),
    (8, 11),
    (2, 5),
)

G1_SKELETON_LABEL = "G1 full body (14 pts)"

# G1_BODY_FRAMES 下标 → COCO-17 body 下标（供 stick figure 预览，与 obs 同拓扑）
_G1_IDX_TO_COCO17: tuple[tuple[int, int], ...] = (
    (2, 5),   # L shoulder
    (3, 7),   # L elbow
    (4, 9),   # L wrist ← palm
    (5, 6),   # R shoulder
    (6, 8),   # R elbow
    (7, 10),  # R wrist
    (8, 11),  # L hip
    (9, 13),  # L knee
    (10, 15), # L ankle ← foot
    (11, 12), # R hip
    (12, 14), # R knee
    (13, 16), # R ankle
)


def g1_kpts_to_coco17(g1_kpts: np.ndarray) -> np.ndarray:
    """G1 14 点 FK → (17,4) COCO body（脸点 conf=0 不画）。"""
    g = np.asarray(g1_kpts, dtype=np.float64).reshape(-1, 4)
    out = np.zeros((17, 4), dtype=np.float64)
    for gi, ci in _G1_IDX_TO_COCO17:
        if gi < g.shape[0]:
            out[ci] = g[gi]
    return out


@dataclass
class G1BodyFK:
    model_path: Path | str

    def __post_init__(self) -> None:
        import mujoco

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(Path(self.model_path).resolve()))
        self.data = mujoco.MjData(self.model)
        self._frames: list[tuple[int, str]] = []
        for name, kind in G1_BODY_FRAMES:
            obj = mujoco.mjtObj.mjOBJ_BODY if kind == "body" else mujoco.mjtObj.mjOBJ_SITE
            fid = int(mujoco.mj_name2id(self.model, obj, name))
            if fid < 0:
                raise ValueError(f"G1 MJCF 缺少 {kind} {name!r}")
            self._frames.append((fid, kind))

    def kpts3d(self, qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        self.data.qvel[:] = 0.0
        self._mujoco.mj_forward(self.model, self.data)
        n = len(self._frames)
        kpts = np.zeros((n, 4), dtype=np.float64)
        for i, (fid, kind) in enumerate(self._frames):
            if kind == "body":
                kpts[i, :3] = self.data.xpos[fid]
            else:
                kpts[i, :3] = self.data.site_xpos[fid]
            kpts[i, 3] = 1.0
        return kpts
