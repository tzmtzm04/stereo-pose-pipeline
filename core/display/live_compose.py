"""display 侧：原始视频 + PoseStore exact match → 可显示 BGR（叠图逻辑外置）。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Protocol

import cv2
import numpy as np

from app.route2.session_config import Route2SessionConfig
from core.display.display_eye import ensure_display_eye_bgr, select_pose_det
from core.display.pose2d_overlay import overlay_pose2d_on_preview
from core.display.pose_store import MatchPolicy, PoseDisplaySample, PoseStore
from orchestration.constants import DEFAULT_DISPLAY_HEIGHT, DEFAULT_DISPLAY_WIDTH

OverlayFn = Callable[
    [np.ndarray, PoseDisplaySample | None, Route2SessionConfig],
    np.ndarray,
]


class OverlayComposer(Protocol):
    def __call__(
        self,
        video_bgr: np.ndarray,
        pose: PoseDisplaySample | None,
        cfg: Route2SessionConfig,
    ) -> np.ndarray: ...


def default_pose_overlay(
    video_bgr: np.ndarray,
    pose: PoseDisplaySample | None,
    cfg: Route2SessionConfig,
) -> np.ndarray:
    """默认 2D 骨架叠图；可替换为自定义 OverlayFn，不改 infer/display 架构。"""
    det = select_pose_det(pose, cfg)
    if not cfg.camera_overlay or det is None:
        return video_bgr
    return overlay_pose2d_on_preview(
        video_bgr,
        det,
        kpts3d=pose.kpts3d if pose is not None else None,
        conf=cfg.conf,
    )


def resolve_display_size(cfg: Route2SessionConfig) -> tuple[int, int]:
    w = cfg.display_width if cfg.display_width > 0 else DEFAULT_DISPLAY_WIDTH
    h = cfg.display_height if cfg.display_height > 0 else (
        DEFAULT_DISPLAY_HEIGHT if cfg.display_width <= 0 else w
    )
    return int(w), int(h)


def _ensure_left_eye_bgr(bgr: np.ndarray) -> np.ndarray:
    """兼容旧名：默认左眼裁切。"""
    h, w = bgr.shape[:2]
    if w > h * 2:
        return bgr[:, : w // 2]
    return bgr


def _ensure_display_eye_bgr(
    bgr: np.ndarray,
    cfg: Route2SessionConfig,
) -> np.ndarray:
    return ensure_display_eye_bgr(bgr, cfg)


def fit_preview(preview: np.ndarray, cw: int, ch: int) -> np.ndarray:
    ih, iw = preview.shape[:2]
    if ih == ch and iw == cw:
        return preview
    scale = min(cw / float(iw), ch / float(ih))
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    resized = cv2.resize(preview, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((ch, cw, 3), (24, 24, 28), np.uint8)
    ox = (cw - nw) // 2
    oy = (ch - nh) // 2
    canvas[oy : oy + nh, ox : ox + nw] = resized
    return canvas


def cover_preview(preview: np.ndarray, cw: int, ch: int) -> np.ndarray:
    """铺满目标矩形（居中裁切），无 letterbox 灰边。"""
    ih, iw = preview.shape[:2]
    if ih == ch and iw == cw:
        return preview
    scale = max(cw / float(iw), ch / float(ih))
    nw = max(1, int(iw * scale))
    nh = max(1, int(ih * scale))
    resized = cv2.resize(preview, (nw, nh), interpolation=cv2.INTER_AREA)
    ox = max(0, (nw - cw) // 2)
    oy = max(0, (nh - ch) // 2)
    return resized[oy : oy + ch, ox : ox + cw].copy()


def match_pose(
    store: PoseStore,
    capture_seq: int,
    *,
    policy: MatchPolicy = "latest_le",
) -> PoseDisplaySample | None:
    return store.match(capture_seq, policy=policy)


def _exact_match_with_wait(
    store: PoseStore,
    capture_seq: int,
    wait_ms: float,
) -> PoseDisplaySample | None:
    """exact match；可选短等 infer 写入 PoseStore。"""
    seq = int(capture_seq)
    pose = match_pose(store, seq, policy="exact")
    if pose is not None or wait_ms <= 0:
        return pose
    deadline = time.monotonic() + float(wait_ms)
    while time.monotonic() < deadline:
        pose = match_pose(store, seq, policy="exact")
        if pose is not None:
            return pose
        time.sleep(0.0002)
    return None


def _pose_usable_for_display(
    pose: PoseDisplaySample | None,
    cfg: Route2SessionConfig,
) -> bool:
    if pose is None:
        return False
    det = select_pose_det(pose, cfg)
    if det is None:
        return False
    kpts = det.get("keypoints")
    if kpts is None:
        return False
    arr = np.asarray(kpts, dtype=np.float32)
    if arr.shape[0] < 5:
        return False
    conf = float(cfg.conf)
    vis = int((arr[:, 2] >= conf).sum())
    if vis < 3:
        return False
    # 全身点（非仅 COCO 0–4 头）：yolo26 等常头点低 conf 但肩/髋可靠
    valid_xy = arr[
        (arr[:, 2] >= conf) & ((arr[:, 0] > 4.0) | (arr[:, 1] > 4.0))
    ]
    return len(valid_xy) >= 2


@dataclass(frozen=True)
class ResolvedDisplayPose:
    pose: PoseDisplaySample | None
    held: bool = False


def resolve_pose_sample_for_display(
    cfg: Route2SessionConfig,
    pose: PoseDisplaySample,
) -> ResolvedDisplayPose:
    """骨架驱动 UI：已有 pose 样本时仅做质量门控。"""
    if _pose_usable_for_display(pose, cfg):
        return ResolvedDisplayPose(pose=pose, held=False)
    return ResolvedDisplayPose(pose=None, held=False)


def resolve_exact_display_pose(
    cfg: Route2SessionConfig,
    store: PoseStore,
    capture_seq: int,
) -> ResolvedDisplayPose:
    """仅 capture_seq 精确对齐且检测有效时返回 pose，否则不叠图。"""
    pose = _exact_match_with_wait(
        store,
        int(capture_seq),
        cfg.display_pose_ui_wait_ms,
    )
    if pose is not None and _pose_usable_for_display(pose, cfg):
        return ResolvedDisplayPose(pose=pose, held=False)
    return ResolvedDisplayPose(pose=None, held=False)


class DisplayPoseResolver:
    """UI 叠图：仅 exact match（pose.capture_seq == 视频 capture_seq）。"""

    def __init__(self, cfg: Route2SessionConfig) -> None:
        self._cfg = cfg

    def resolve(
        self,
        store: PoseStore,
        capture_seq: int,
        *,
        exact_only: bool = True,
    ) -> ResolvedDisplayPose:
        _ = exact_only
        return resolve_exact_display_pose(self._cfg, store, capture_seq)


def compose_display_frame(
    cfg: Route2SessionConfig,
    video_bgr: np.ndarray | None,
    pose: PoseDisplaySample | None,
    *,
    overlay_fn: OverlayFn | None = None,
    out_size: tuple[int, int] | None = None,
    fill: str = "fit",
) -> np.ndarray:
    """原始 BGR + 可选 pose → 适配窗口尺寸的 composite（未含状态栏）。

    fill: \"fit\" 留边；\"cover\" 铺满裁切（嵌入贴洞用，避免外围灰框）。
    """
    if out_size is not None:
        cw, ch = int(out_size[0]), int(out_size[1])
    else:
        cw, ch = resolve_display_size(cfg)
    cw = max(1, cw)
    ch = max(1, ch)
    if video_bgr is None:
        return np.full((ch, cw, 3), (32, 32, 36), np.uint8)
    video_bgr = _ensure_display_eye_bgr(video_bgr, cfg)
    fn = overlay_fn or default_pose_overlay
    vis = fn(video_bgr, pose, cfg)
    if fill == "cover":
        return cover_preview(vis, cw, ch)
    return fit_preview(vis, cw, ch)
