"""离线纯 3D 骨架 preview（无背景视频，复用 skeleton3d_view）。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.display.skeleton3d_layout import (
    Skeleton3DLayoutMode,
    resolve_skeleton3d_render_spec,
)
from core.display.skeleton3d_view import (
    DEFAULT_VOLUME_5M,
    ProjectionMode,
    Skeleton3DRotation,
    Volume3DBox,
    last_skeleton_render_tag,
    render_skeleton3d_bgr,
    reset_canvas_buffers,
)
from core.offline.offline_preview import OfflinePoseRow
from core.offline.two_stage.coco_wholebody_layout import FEET_JOINT_INDICES

STATUS_H = 28
TIMELINE_H = 40
_TIMELINE_MARGIN = 16


@dataclass
class PoseTimeline:
    """按时间/帧索引拖动。"""

    count: int
    index: int = 0
    fps: float = 60.0
    dragging: bool = False

    @property
    def duration_sec(self) -> float:
        if self.count <= 1:
            return 0.0
        return (self.count - 1) / max(float(self.fps), 1e-6)

    @property
    def time_sec(self) -> float:
        return self.index / max(float(self.fps), 1e-6)

    def clamp_index(self) -> None:
        if self.count <= 0:
            self.index = 0
            return
        self.index = int(np.clip(self.index, 0, self.count - 1))

    def seek_fraction(self, frac: float) -> None:
        if self.count <= 1:
            self.index = 0
            return
        self.index = int(round(float(np.clip(frac, 0.0, 1.0)) * (self.count - 1)))

    def seek_from_x(self, x: int, width: int) -> None:
        track_w = max(1, int(width) - 2 * _TIMELINE_MARGIN)
        frac = (int(x) - _TIMELINE_MARGIN) / float(track_w)
        self.seek_fraction(frac)

    def step(self, delta: int) -> None:
        self.index += int(delta)
        self.clamp_index()


@dataclass
class Skeleton3DPreviewUI:
    window: str
    rotation: Skeleton3DRotation
    timeline: PoseTimeline
    skeleton_w: int
    skeleton_h: int
    _rot_drag: bool = False
    _last_x: int = 0
    _last_y: int = 0

    @property
    def skeleton_y0(self) -> int:
        return STATUS_H

    @property
    def timeline_y0(self) -> int:
        return STATUS_H + int(self.skeleton_h)

    @property
    def total_h(self) -> int:
        return STATUS_H + int(self.skeleton_h) + TIMELINE_H

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        in_timeline = y >= self.timeline_y0
        in_skeleton = self.skeleton_y0 <= y < self.timeline_y0

        if in_timeline:
            if event == cv2.EVENT_LBUTTONDOWN:
                self.timeline.dragging = True
                self.timeline.seek_from_x(x, self.skeleton_w)
            elif event == cv2.EVENT_MOUSEMOVE and self.timeline.dragging:
                self.timeline.seek_from_x(x, self.skeleton_w)
            elif event == cv2.EVENT_LBUTTONUP:
                self.timeline.dragging = False
            return
        elif event == cv2.EVENT_LBUTTONUP:
            self.timeline.dragging = False
            self._rot_drag = False
            return

        if not in_skeleton:
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            self._rot_drag = True
            self._last_x, self._last_y = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            self._rot_drag = False
        elif event == cv2.EVENT_MOUSEMOVE and self._rot_drag:
            dx = x - self._last_x
            dy = y - self._last_y
            if self.rotation.y_up_locked:
                self.rotation.apply_drag(dx, 0)
            else:
                self.rotation.apply_drag(dx, dy)
            self._last_x, self._last_y = x, y


def compose_skeleton3d_frame(
    row: OfflinePoseRow | None,
    *,
    rotation: Skeleton3DRotation,
    conf: float = 0.25,
    width: int = 720,
    height: int = 720,
    ignore_conf: bool = True,
    layout_mode: Skeleton3DLayoutMode = "auto",
    show_feet: bool = False,
    projection_mode: ProjectionMode = "volume",
    volume_box: Volume3DBox | None = None,
) -> tuple[np.ndarray, str]:
    """kpts3d → 黑底网格 + 3D 骨架（自动识别 17 / 23 / 133）。"""
    kpts = row.kpts3d if row is not None else None
    spec = resolve_skeleton3d_render_spec(
        kpts, mode=layout_mode, include_feet=show_feet
    )
    hidden = frozenset() if show_feet else frozenset(FEET_JOINT_INDICES)
    vol = volume_box or DEFAULT_VOLUME_5M
    canvas = render_skeleton3d_bgr(
        kpts,
        width=int(width),
        height=int(height),
        conf=float(conf),
        rotation=rotation,
        ignore_conf=ignore_conf,
        skeleton_edges=spec.edges,
        layout=spec.layout,
        label=spec.label,
        exclude_joint_indices=hidden,
        projection_mode=projection_mode,
        volume_box=vol,
    )
    return canvas, spec.label


def build_skeleton3d_status_line(
    *,
    capture_seq: int,
    infer_seq: int,
    tri_ok: bool,
    frame_idx: int,
    fps: float,
    render_tag: str,
    time_sec: float,
    duration_sec: float,
    paused: bool,
    layout_label: str = "",
    proj_label: str = "",
) -> str:
    tri_s = "tri_ok" if tri_ok else "tri_fail"
    pause_s = "PAUSE" if paused else "PLAY"
    layout_s = f" {layout_label}" if layout_label else ""
    proj_s = f" {proj_label}" if proj_label else ""
    return (
        f"offline 3D {pause_s}{layout_s}{proj_s} {time_sec:.2f}/{duration_sec:.2f}s "
        f"seq={capture_seq} n={frame_idx} {tri_s} tag={render_tag} "
        f"fps={fps:.1f}"
    )


def _draw_timeline_bar(
    canvas: np.ndarray,
    timeline: PoseTimeline,
    y0: int,
    width: int,
) -> None:
    y_mid = y0 + TIMELINE_H // 2
    x0 = _TIMELINE_MARGIN
    x1 = width - _TIMELINE_MARGIN
    cv2.rectangle(canvas, (x0, y_mid - 6), (x1, y_mid + 6), (52, 52, 62), -1)
    cv2.rectangle(canvas, (x0, y_mid - 6), (x1, y_mid + 6), (80, 80, 92), 1)

    if timeline.count > 1:
        frac = timeline.index / (timeline.count - 1)
        px = int(x0 + frac * (x1 - x0))
        cv2.line(canvas, (x0, y_mid), (px, y_mid), (0, 180, 120), 3, cv2.LINE_AA)
        cv2.circle(canvas, (px, y_mid), 9, (0, 210, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, y_mid), 9, (240, 240, 240), 1, cv2.LINE_AA)

    hint = "拖动时间轴 seek | 骨架区水平拖=绕Y轴左右转"
    cv2.putText(
        canvas,
        hint,
        (x0, y0 + TIMELINE_H - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (140, 140, 160),
        1,
        cv2.LINE_AA,
    )


def build_skeleton3d_display(
    skeleton_bgr: np.ndarray,
    *,
    status: str,
    timeline: PoseTimeline,
) -> np.ndarray:
    h, w = skeleton_bgr.shape[:2]
    canvas = np.full((STATUS_H + h + TIMELINE_H, w, 3), (24, 24, 28), np.uint8)
    cv2.putText(
        canvas,
        status,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    canvas[STATUS_H : STATUS_H + h, :] = skeleton_bgr
    _draw_timeline_bar(canvas, timeline, STATUS_H + h, w)
    return canvas


def show_skeleton3d_display(window: str, display: np.ndarray) -> None:
    cv2.imshow(window, display)


def setup_skeleton3d_window(ui: Skeleton3DPreviewUI) -> None:
    reset_canvas_buffers()
    cv2.namedWindow(ui.window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ui.window, int(ui.skeleton_w), int(ui.total_h))
    cv2.setMouseCallback(ui.window, ui.on_mouse)


def teardown_skeleton3d_window(window: str) -> None:
    cv2.destroyWindow(window)
    reset_canvas_buffers()
