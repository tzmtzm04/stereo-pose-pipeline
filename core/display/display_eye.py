"""实时 display 单眼选择（左/右）。"""

from __future__ import annotations

from typing import Literal

import numpy as np

from app.route2.session_config import Route2SessionConfig
from core.display.pose_store import PoseDisplaySample

DisplayEye = Literal["left", "right"]


def normalize_display_eye(eye: str) -> DisplayEye:
    key = str(eye).strip().lower()
    if key in ("right", "r", "1"):
        return "right"
    return "left"


def display_eye_name(cfg: Route2SessionConfig) -> DisplayEye:
    return normalize_display_eye(str(getattr(cfg, "display_eye", "left")))


def display_eye_is_right(cfg: Route2SessionConfig) -> bool:
    return display_eye_name(cfg) == "right"


def sbs_crop_x_for_eye(full_w: int, eye: str) -> int:
    ew = max(1, int(full_w) // 2)
    return ew if normalize_display_eye(eye) == "right" else 0


def sbs_crop_x(full_w: int, cfg: Route2SessionConfig) -> int:
    ew = max(1, int(full_w) // 2)
    return ew if display_eye_is_right(cfg) else 0


def select_pose_det(
    pose: PoseDisplaySample | object | None,
    cfg: Route2SessionConfig,
) -> dict | None:
    if pose is None:
        return None
    key = "det_right" if display_eye_is_right(cfg) else "det_left"
    return getattr(pose, key, None)


def ensure_display_eye_bgr(
    bgr: np.ndarray,
    cfg: Route2SessionConfig,
) -> np.ndarray:
    """若误传入全幅 SBS，按 cfg.display_eye 裁单眼。"""
    h, w = bgr.shape[:2]
    if w > h * 2:
        ew = w // 2
        if display_eye_is_right(cfg):
            return bgr[:, ew : ew + ew]
        return bgr[:, :ew]
    return bgr


def gl_sbs_eye_mode(cfg: Route2SessionConfig, *, is_sbs: bool) -> int:
    """0=全幅，1=左眼，2=右眼。"""
    if not is_sbs:
        return 0
    return 2 if display_eye_is_right(cfg) else 1
