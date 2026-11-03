"""SOMA 骨架数据（JSON）加载。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_JSON = Path(__file__).resolve().parents[2] / "data" / "soma_skeleton.json"


@dataclass(frozen=True)
class SomaSkeleton:
    num_joints: int
    joint_names: tuple[str, ...]
    parent: np.ndarray  # (78,) int
    rest_local: np.ndarray  # (78, 3) float32
    bone_lengths: np.ndarray  # (78,) float32
    coco_to_soma: np.ndarray  # (17,) int
    coco_offset: np.ndarray  # (17, 3) float32
    joint_limits: np.ndarray  # (78, 6) float32
    spine_chain: tuple[int, ...]
    limb_chains: dict[str, tuple[int, ...]]
    twist_joints: tuple[int, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SomaSkeleton:
        limb_raw = data.get("limb_chains") or {}
        limb_chains = {
            str(k): tuple(int(x) for x in v) for k, v in limb_raw.items()
        }
        return cls(
            num_joints=int(data["num_joints"]),
            joint_names=tuple(str(x) for x in data["joint_names"]),
            parent=np.asarray(data["parent"], dtype=np.int32),
            rest_local=np.asarray(data["rest_local"], dtype=np.float32),
            bone_lengths=np.asarray(data["bone_lengths"], dtype=np.float32),
            coco_to_soma=np.asarray(data["coco_to_soma"], dtype=np.int32),
            coco_offset=np.asarray(data["coco_offset_in_soma"], dtype=np.float32),
            joint_limits=np.asarray(data["joint_limits"], dtype=np.float32),
            spine_chain=tuple(int(x) for x in data.get("spine_chain", (1, 2, 3, 4, 5))),
            limb_chains=limb_chains,
            twist_joints=tuple(int(x) for x in data.get("twist_joints", (16, 43, 71, 77))),
        )

    def children_of(self, joint_idx: int) -> list[int]:
        return [i for i in range(self.num_joints) if int(self.parent[i]) == joint_idx]

    def pose_joint_index(self, joint_idx: int) -> int:
        """关节索引 → pose 轴角数组下标（joint 0 无 pose）。"""
        if joint_idx <= 0:
            raise ValueError("Root 无 pose DOF")
        return joint_idx - 1


def load_soma_skeleton(path: Path | str | None = None) -> SomaSkeleton:
    p = Path(path).expanduser().resolve() if path else _DEFAULT_JSON
    if not p.is_file():
        raise FileNotFoundError(
            f"缺少骨架 JSON: {p}（运行 scripts/geometry/generate_soma_skeleton.py）"
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    return SomaSkeleton.from_dict(data)


@lru_cache(maxsize=1)
def default_soma_skeleton() -> SomaSkeleton:
    return load_soma_skeleton()


def reload_default_soma_skeleton() -> SomaSkeleton:
    default_soma_skeleton.cache_clear()
    return default_soma_skeleton()
