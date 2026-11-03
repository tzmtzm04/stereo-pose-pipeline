"""离线 2D 叠图 preview：可拖拽时间轴 seek。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from contracts.session_bundle import SessionBundle
from core.geometry.calib_model import StereoCalibration
from core.offline.offline_preview import (
    OfflinePoseRow,
    compose_offline_left_eye_frame,
    compose_offline_right_eye_frame,
)
from core.offline.offline_skeleton3d_preview import (
    STATUS_H,
    TIMELINE_H,
    PoseTimeline,
    _draw_timeline_bar,
)
from core.offline.session_decoder import SessionIndexRow, decode_session_frame


@dataclass
class Offline2DScrubUI:
    window: str
    timeline: PoseTimeline
    video_w: int
    video_h: int

    @property
    def video_y0(self) -> int:
        return STATUS_H

    @property
    def timeline_y0(self) -> int:
        return STATUS_H + int(self.video_h)

    @property
    def total_h(self) -> int:
        return STATUS_H + int(self.video_h) + TIMELINE_H

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if y < self.timeline_y0:
            if event == cv2.EVENT_LBUTTONUP:
                self.timeline.dragging = False
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            self.timeline.dragging = True
            self.timeline.seek_from_x(x, self.video_w)
        elif event == cv2.EVENT_MOUSEMOVE and self.timeline.dragging:
            self.timeline.seek_from_x(x, self.video_w)
        elif event == cv2.EVENT_LBUTTONUP:
            self.timeline.dragging = False


_STEREO_LABEL_H = 22
_STEREO_DIVIDER_W = 4


def compose_offline_stereo_frames(
    left_bgr: np.ndarray,
    right_bgr: np.ndarray,
    pose: OfflinePoseRow | None,
    *,
    calib: StereoCalibration | None,
    conf: float,
    overlay: bool,
    allow_proj3d: bool,
    prefer_proj3d: bool = False,
    coords_rectified: bool = False,
    rectifier: object | None = None,
    display_width: int,
    display_height: int,
) -> tuple[np.ndarray, str]:
    """左右目 BGR + 2D 骨架并排；返回 (bgr, overlay_source)。"""
    divider = _STEREO_DIVIDER_W
    panel_w = max(1, (int(display_width) - divider) // 2)
    panel_h = int(display_height)

    left_comp, left_src = compose_offline_left_eye_frame(
        left_bgr,
        pose,
        calib=calib,
        conf=conf,
        overlay=overlay,
        allow_proj3d=allow_proj3d,
        prefer_proj3d=prefer_proj3d,
        coords_rectified=coords_rectified,
        rectifier=rectifier,
        display_width=panel_w,
        display_height=panel_h,
    )
    right_comp, right_src = compose_offline_right_eye_frame(
        right_bgr,
        pose,
        conf=conf,
        overlay=overlay,
        display_width=panel_w,
        display_height=panel_h,
    )

    def _label_panel(img: np.ndarray, text: str) -> np.ndarray:
        bar = np.zeros((_STEREO_LABEL_H, panel_w, 3), np.uint8)
        cv2.putText(
            bar,
            text,
            (8, 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 220, 255),
            1,
            cv2.LINE_AA,
        )
        body_h = max(1, panel_h - _STEREO_LABEL_H)
        body = cv2.resize(img, (panel_w, body_h), interpolation=cv2.INTER_AREA)
        return np.vstack([bar, body])

    left_l = _label_panel(left_comp, "left +2D")
    right_l = _label_panel(right_comp, "right +2D")
    sep = np.full((panel_h, divider, 3), (48, 48, 52), np.uint8)
    composite = np.hstack([left_l, sep, right_l])
    if left_src == "none" and right_src == "none":
        src = "none"
    elif left_src == right_src:
        src = left_src
    else:
        src = f"L:{left_src}|R:{right_src}"
    return composite, src


def compose_offline_stereo_overlay(
    bundle: SessionBundle,
    row: SessionIndexRow,
    pose: OfflinePoseRow | None,
    *,
    calib: StereoCalibration | None,
    conf: float,
    overlay: bool,
    allow_proj3d: bool,
    prefer_proj3d: bool = False,
    coords_rectified: bool = False,
    rectifier: object | None = None,
    display_width: int,
    display_height: int,
) -> tuple[np.ndarray, str]:
    """Session 索引行 → 左右目并排叠图。"""
    frame = decode_session_frame(bundle, row)
    if frame is None:
        canvas = np.full(
            (int(display_height), int(display_width), 3), (32, 32, 36), np.uint8
        )
        return canvas, "none"
    return compose_offline_stereo_frames(
        frame.left_bgr,
        frame.right_bgr,
        pose,
        calib=calib,
        conf=conf,
        overlay=overlay,
        allow_proj3d=allow_proj3d,
        prefer_proj3d=prefer_proj3d,
        coords_rectified=coords_rectified,
        rectifier=rectifier,
        display_width=display_width,
        display_height=display_height,
    )


def compose_offline_eye_overlay(
    bundle: SessionBundle,
    row: SessionIndexRow,
    pose: OfflinePoseRow | None,
    *,
    eye: str,
    calib: StereoCalibration | None,
    conf: float,
    overlay: bool,
    allow_proj3d: bool,
    prefer_proj3d: bool = False,
    coords_rectified: bool = False,
    rectifier: object | None = None,
    display_width: int,
    display_height: int,
) -> tuple[np.ndarray, str]:
    if eye == "both":
        return compose_offline_stereo_overlay(
            bundle,
            row,
            pose,
            calib=calib,
            conf=conf,
            overlay=overlay,
            allow_proj3d=allow_proj3d,
            prefer_proj3d=prefer_proj3d,
            coords_rectified=coords_rectified,
            rectifier=rectifier,
            display_width=display_width,
            display_height=display_height,
        )
    frame = decode_session_frame(bundle, row)
    if frame is None:
        canvas = np.full((int(display_height), int(display_width), 3), (32, 32, 36), np.uint8)
        return canvas, "none"
    eye_bgr = frame.right_bgr if eye == "right" else frame.left_bgr
    if eye == "right":
        return compose_offline_right_eye_frame(
            eye_bgr,
            pose,
            conf=conf,
            overlay=overlay,
            display_width=display_width,
            display_height=display_height,
        )
    return compose_offline_left_eye_frame(
        eye_bgr,
        pose,
        calib=calib,
        conf=conf,
        overlay=overlay,
        allow_proj3d=allow_proj3d,
        prefer_proj3d=prefer_proj3d,
        coords_rectified=coords_rectified,
        rectifier=rectifier,
        display_width=display_width,
        display_height=display_height,
    )


def build_offline_scrub_status_line(
    *,
    capture_seq: int,
    infer_seq: int,
    tri_ok: bool,
    frame_idx: int,
    fps: float,
    overlay: bool,
    overlay_source: str,
    eye: str,
    time_sec: float,
    duration_sec: float,
    paused: bool,
) -> str:
    tri_s = "tri_ok" if tri_ok else "tri_fail"
    eye_s = "stereo" if eye == "both" else ("right" if eye == "right" else "left")
    pause_s = "PAUSE" if paused else "PLAY"
    if not overlay:
        ov = "raw"
    elif overlay_source == "det2d":
        ov = "+2D"
    elif overlay_source == "det2d(distort)":
        ov = "+2D(dist)"
    elif overlay_source == "proj3d":
        ov = "+2D(proj)"
    else:
        ov = "no_pose"
    return (
        f"offline {eye_s} {pause_s} {time_sec:.2f}/{duration_sec:.2f}s "
        f"seq={capture_seq} n={frame_idx} infer={infer_seq} {tri_s} "
        f"fps={fps:.1f} {ov} [拖动时间轴|←→|space|q]"
    )


def build_offline_scrub_display(
    composite_bgr: np.ndarray,
    *,
    status: str,
    timeline: PoseTimeline,
) -> np.ndarray:
    h, w = composite_bgr.shape[:2]
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
    canvas[STATUS_H : STATUS_H + h, :] = composite_bgr
    _draw_timeline_bar(canvas, timeline, STATUS_H + h, w)
    return canvas


def setup_offline_scrub_window(ui: Offline2DScrubUI) -> None:
    cv2.namedWindow(ui.window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ui.window, int(ui.video_w), int(ui.total_h))
    cv2.setMouseCallback(ui.window, ui.on_mouse)


def teardown_offline_scrub_window(window: str) -> None:
    cv2.destroyWindow(window)
