"""COCO / SMPL / SOMA 关节映射表加载。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_JSON = Path(__file__).resolve().parents[2] / "data" / "body_joint_maps.json"


@dataclass(frozen=True)
class BodyJointMaps:
    coco17_names: tuple[str, ...]
    smpl24_names: tuple[str, ...]
    coco_to_smpl_direct: dict[int, int]
    coco_to_smpl_derived: dict[int, dict[str, Any]]
    opencv_lr_swap_pairs: tuple[tuple[int, int], ...]
    smpl24_to_soma_joint: np.ndarray  # (24,) int
    smpl24_to_soma_offset: np.ndarray  # (24, 3) float32
    spine_smpl_indices: tuple[int, ...]
    spine_soma_indices: tuple[int, ...]
    limb_smpl_triplets: dict[str, tuple[int, ...]]
    limb_soma_chains: dict[str, tuple[int, ...]]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BodyJointMaps:
        c2s = data["coco_to_smpl24"]
        s2o = data["smpl24_to_soma"]
        direct_raw = c2s.get("direct") or {}
        direct = {int(k): int(v) for k, v in direct_raw.items()}
        derived_raw = c2s.get("derived") or {}
        derived = {int(k): v for k, v in derived_raw.items()}
        swap = tuple(tuple(int(x) for x in p) for p in c2s.get("opencv_lr_swap_pairs", ()))
        limb_smpl = {
            str(k): tuple(int(x) for x in v)
            for k, v in (s2o.get("limb_smpl_triplets") or {}).items()
        }
        limb_soma = {
            str(k): tuple(int(x) for x in v)
            for k, v in (s2o.get("limb_soma_chains") or {}).items()
        }
        return cls(
            coco17_names=tuple(str(x) for x in data["coco17_names"]),
            smpl24_names=tuple(str(x) for x in data["smpl24_names"]),
            coco_to_smpl_direct=direct,
            coco_to_smpl_derived=derived,
            opencv_lr_swap_pairs=swap,
            smpl24_to_soma_joint=np.asarray(s2o["joint_index"], dtype=np.int32),
            smpl24_to_soma_offset=np.asarray(s2o["offset_m"], dtype=np.float32),
            spine_smpl_indices=tuple(int(x) for x in s2o["spine_smpl_indices"]),
            spine_soma_indices=tuple(int(x) for x in s2o["spine_soma_indices"]),
            limb_smpl_triplets=limb_smpl,
            limb_soma_chains=limb_soma,
        )


def load_body_joint_maps(path: Path | str | None = None) -> BodyJointMaps:
    p = Path(path).expanduser().resolve() if path else _DEFAULT_JSON
    if not p.is_file():
        raise FileNotFoundError(f"缺少关节映射 JSON: {p}")
    return BodyJointMaps.from_dict(json.loads(p.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def default_body_joint_maps() -> BodyJointMaps:
    return load_body_joint_maps()
