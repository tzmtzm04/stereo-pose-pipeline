"""IK 求解器工厂（DLS / mink）。"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from core.mujoco_ik.jacobian_ik import JacobianIKConfig, JacobianIKSolver
from core.mujoco_ik.layout import IKSiteSpec
from core.mujoco_ik.mink_ik import MinkIKConfig, MinkIKSolver

IKBackend = Literal["dls", "mink", "jacobian_dls"]


class IKSolver(Protocol):
    joint_names: list[str]

    def reset(self) -> None: ...

    def solve(
        self,
        targets: dict[str, np.ndarray | None],
        *,
        qpos_init: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, int]: ...


def normalize_ik_backend(name: str) -> IKBackend:
    key = str(name).strip().lower().replace("-", "_")
    if key in ("dls", "jacobian", "jacobian_dls", "default"):
        return "dls"
    if key == "mink":
        return "mink"
    raise ValueError(f"未知 IK backend: {name!r}（dls | mink）")


def ik_method_label(backend: IKBackend) -> str:
    return "mink" if backend == "mink" else "jacobian_dls"


def make_ik_solver(
    model_path: Path | str,
    *,
    backend: IKBackend = "dls",
    dls_cfg: JacobianIKConfig | None = None,
    mink_cfg: MinkIKConfig | None = None,
    site_specs: tuple[IKSiteSpec, ...] | None = None,
    init_from_keyframe: bool = False,
    root_anchor_sites: tuple[str, ...] | None = None,
    frozen_joint_names: tuple[str, ...] | None = None,
) -> IKSolver:
    if backend == "mink":
        return MinkIKSolver(
            model_path,
            cfg=mink_cfg,
            site_specs=site_specs,
            init_from_keyframe=init_from_keyframe,
            root_anchor_sites=root_anchor_sites,
            frozen_joint_names=frozen_joint_names,
        )
    return JacobianIKSolver(
        model_path,
        cfg=dls_cfg,
        site_specs=site_specs,
        init_from_keyframe=init_from_keyframe,
    )


def default_min_targets(
    backend: IKBackend,
    *,
    dls_cfg: JacobianIKConfig | None = None,
    mink_cfg: MinkIKConfig | None = None,
) -> int:
    if backend == "mink":
        return (mink_cfg or MinkIKConfig()).min_targets
    return (dls_cfg or JacobianIKConfig()).min_targets
