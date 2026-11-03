"""OpenCV 预览 UI（主线程）— 左眼原图。"""

from __future__ import annotations

import os
import threading

import cv2
import numpy as np

from app.route2.session_config import Route2SessionConfig
from core.display.display_eye import display_eye_name, ensure_display_eye_bgr, select_pose_det
from core.display.pose2d_overlay import overlay_pose2d_on_preview
from domain.l3.runtime import L3Runtime
from orchestration.types import DisplayPacket
from orchestration.constants import (
    DEFAULT_DISPLAY_HEIGHT,
    DEFAULT_DISPLAY_WIDTH,
    DISPLAY_WINDOW,
    STATUS_BAND_H,
)

_display_window_open = False
_display_first_frame_shown = False
_window_thread_started = False
_window_size: tuple[int, int] | None = None
_vis_buffer: np.ndarray | None = None
_status_bar_fill = np.uint8(24)
_placeholder: np.ndarray | None = None
# fall_risk 嵌套：无状态条、初始屏外、尽量置顶（盖住 Firefox kiosk）
_embedded_preview_mode = False
_STAGING_ORIGIN = (-8192, -8192)
# Host 只排队几何；必须在 UI 线程 apply（OpenCV 非线程安全）
_embed_lock = threading.Lock()
_pending_embed: tuple[int, int, int, int, bool] | None = None
_applied_embed: tuple[int, int, int, int, bool] | None = None


def set_embedded_preview_mode(enabled: bool) -> None:
    global _embedded_preview_mode
    _embedded_preview_mode = bool(enabled)


def embedded_canvas_size() -> tuple[int, int] | None:
    """kiosk 已贴洞时返回洞尺寸；否则 None（仍用 cfg.display_*）。"""
    if not _embedded_preview_mode:
        return None
    emb = _applied_embed
    if emb is None or not emb[4]:
        return None
    return int(emb[2]), int(emb[3])


def display_window_is_open() -> bool:
    return bool(_display_window_open)


def align_display_window(
    x: int,
    y: int,
    width: int,
    height: int,
    *,
    mapped: bool = True,
) -> tuple[int, int, int, int]:
    """Host 线程：只排队几何，由 UI 线程 apply_pending_display_embed。"""
    global _pending_embed
    w = max(64, int(width))
    h = max(64, int(height))
    if mapped:
        geom = (int(x), int(y), w, h)
    else:
        geom = (_STAGING_ORIGIN[0], _STAGING_ORIGIN[1], w, h)
    with _embed_lock:
        _pending_embed = (geom[0], geom[1], geom[2], geom[3], bool(mapped))
    return geom


def hide_display_window() -> None:
    align_display_window(
        _STAGING_ORIGIN[0],
        _STAGING_ORIGIN[1],
        64,
        64,
        mapped=False,
    )


_embed_x11_logged = False


def apply_pending_display_embed() -> bool:
    """UI 线程：应用 Host 排队的嵌套几何。返回是否发生 MoveResize。"""
    global _pending_embed, _applied_embed, _window_size, _embed_x11_logged
    with _embed_lock:
        pending = _pending_embed
        if pending is None:
            return False
        _pending_embed = None
    if not _display_window_open:
        with _embed_lock:
            # 窗未开：放回队列，下一帧再试
            if _pending_embed is None:
                _pending_embed = pending
        return False
    x, y, w, h, mapped = pending
    key = (x, y, w, h, mapped)
    if _applied_embed == key:
        return False
    try:
        # Qt TOPMOST 压不住 Firefox --kiosk；用 X11 override_redirect（同 GL 嵌套）
        from platform.jetson.x11_overlay import place_override_redirect

        ok = place_override_redirect(
            DISPLAY_WINDOW, x, y, w, h, mapped=mapped
        )
        if not ok:
            # 窗名尚未出现：放回队列下一帧再试，勿标记 applied
            with _embed_lock:
                if _pending_embed is None:
                    _pending_embed = pending
            try:
                cv2.resizeWindow(DISPLAY_WINDOW, w, h)
                cv2.moveWindow(DISPLAY_WINDOW, x, y)
            except Exception:
                pass
            return False
        if not _embed_x11_logged and mapped:
            _embed_x11_logged = True
            print(
                f"fall_risk OpenCV 窗 X11 override_redirect "
                f"{w}x{h}+{x}+{y}（盖住 Firefox kiosk）",
                flush=True,
            )
        _window_size = (w, h)
        _applied_embed = key
        return True
    except Exception as exc:
        if not _embed_x11_logged:
            _embed_x11_logged = True
            print(f"WARN: X11 embed 失败，回退 moveWindow: {exc}", flush=True)
        try:
            cv2.resizeWindow(DISPLAY_WINDOW, w, h)
            cv2.moveWindow(DISPLAY_WINDOW, x, y)
            _raise_display_window()
            _window_size = (w, h)
            _applied_embed = key
            return True
        except Exception:
            return False


def _raise_display_window() -> None:
    try:
        prop = getattr(cv2, "WND_PROP_TOPMOST", None)
        if prop is not None:
            cv2.setWindowProperty(DISPLAY_WINDOW, prop, 1)
    except Exception:
        pass
    try:
        cv2.setWindowTitle(DISPLAY_WINDOW, DISPLAY_WINDOW)
    except Exception:
        pass


def check_display_env(no_display: bool) -> None:
    if no_display:
        return
    if not os.environ.get("DISPLAY"):
        print(
            "警告: DISPLAY 未设置，OpenCV 窗口不会出现。本机桌面请: export DISPLAY=:0",
            flush=True,
        )


def teardown_display_window() -> None:
    global _display_window_open, _display_first_frame_shown, _window_thread_started
    global _window_size, _vis_buffer, _placeholder, _embedded_preview_mode
    global _pending_embed, _applied_embed, _embed_x11_logged
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        from platform.jetson.x11_overlay import clear_prepared

        clear_prepared(DISPLAY_WINDOW)
    except Exception:
        pass
    _display_window_open = False
    _display_first_frame_shown = False
    _window_thread_started = False
    _window_size = None
    _vis_buffer = None
    _placeholder = None
    _embedded_preview_mode = False
    _embed_x11_logged = False
    with _embed_lock:
        _pending_embed = None
        _applied_embed = None


def resolve_display_size(cfg: Route2SessionConfig) -> tuple[int, int]:
    w = cfg.display_width if cfg.display_width > 0 else DEFAULT_DISPLAY_WIDTH
    h = cfg.display_height if cfg.display_height > 0 else (
        DEFAULT_DISPLAY_HEIGHT if cfg.display_width <= 0 else w
    )
    return int(w), int(h)


def show_startup_splash(message: str, *, no_display: bool) -> None:
    if no_display:
        return
    w, h = DEFAULT_DISPLAY_WIDTH, 420
    canvas = np.full((h, w, 3), (28, 28, 32), np.uint8)
    lines = [
        "Route2 启动中...",
        message,
        "",
        f"默认预览: {DEFAULT_DISPLAY_WIDTH}x{DEFAULT_DISPLAY_HEIGHT} 单眼原图",
        "按 q 退出",
    ]
    for i, line in enumerate(lines):
        color = (0, 220, 120) if i == 0 else (180, 180, 200)
        cv2.putText(
            canvas,
            line,
            (20, 48 + i * 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55 if i == 0 else 0.48,
            color,
            1,
            cv2.LINE_AA,
        )
    _ensure_window(w, h)
    cv2.imshow(DISPLAY_WINDOW, canvas)
    cv2.waitKey(30)


def _ensure_window(width: int, height: int) -> None:
    global _display_window_open, _window_size, _window_thread_started
    if not _window_thread_started:
        try:
            cv2.startWindowThread()
        except Exception:
            pass
        _window_thread_started = True
    if not _display_window_open:
        flags = cv2.WINDOW_NORMAL
        # Qt：去掉 toolbar/statusbar，避免嵌洞外围多一圈控件框
        gui_normal = getattr(cv2, "WINDOW_GUI_NORMAL", 0)
        if gui_normal:
            flags |= int(gui_normal)
        cv2.namedWindow(DISPLAY_WINDOW, flags)
        if _embedded_preview_mode:
            # Host 负责最终几何；先屏外，避免闪在桌面中央
            try:
                cv2.setWindowProperty(
                    DISPLAY_WINDOW,
                    cv2.WND_PROP_ASPECT_RATIO,
                    cv2.WINDOW_FREERATIO,
                )
            except Exception:
                pass
            cv2.resizeWindow(DISPLAY_WINDOW, width, height)
            cv2.moveWindow(DISPLAY_WINDOW, _STAGING_ORIGIN[0], _STAGING_ORIGIN[1])
            _raise_display_window()
        else:
            cv2.resizeWindow(DISPLAY_WINDOW, width, height)
            try:
                cv2.setWindowProperty(
                    DISPLAY_WINDOW,
                    cv2.WND_PROP_ASPECT_RATIO,
                    cv2.WINDOW_KEEPRATIO,
                )
            except Exception:
                pass
        _window_size = (width, height)
        _display_window_open = True
    elif (not _embedded_preview_mode) and _window_size != (width, height):
        # 嵌套模式：几何由 Host align，避免与 UI 线程抢 resize
        cv2.resizeWindow(DISPLAY_WINDOW, width, height)
        _window_size = (width, height)


def _ensure_vis_buffer(width: int, canvas_h: int) -> np.ndarray:
    global _vis_buffer
    total_h = canvas_h + STATUS_BAND_H
    if (
        _vis_buffer is None
        or _vis_buffer.shape[0] != total_h
        or _vis_buffer.shape[1] != width
    ):
        _vis_buffer = np.empty((total_h, width, 3), dtype=np.uint8)
        _vis_buffer[canvas_h:, :, :] = _status_bar_fill
    return _vis_buffer


def _canvas_size(cfg: Route2SessionConfig) -> tuple[int, int]:
    return resolve_display_size(cfg)


def _placeholder_canvas(w: int, h: int) -> np.ndarray:
    global _placeholder
    if _placeholder is None or _placeholder.shape[:2] != (h, w):
        _placeholder = np.full((h, w, 3), (32, 32, 36), np.uint8)
        cv2.putText(
            _placeholder,
            "no frame",
            (w // 2 - 60, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (100, 100, 120),
            2,
            cv2.LINE_AA,
        )
    return _placeholder


def _fit_preview(preview: np.ndarray, cw: int, ch: int) -> np.ndarray:
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


def build_status_line(
    cfg: Route2SessionConfig,
    l3: L3Runtime | None,
    *,
    frame_idx: int,
    capture_seq: int,
    infer_seq: int,
    tri_ok: bool,
) -> str:
    status = "3D OK" if tri_ok else "no pose"
    parts = [status, f"eye={display_eye_name(cfg)}", f"cap={capture_seq}", f"infer={infer_seq}"]
    if l3 is not None:
        parts.append(l3.display_status_suffix())
    parts.append("q quit")
    parts.append(f"f={frame_idx}")
    return "  ".join(parts)


def render_frame(
    cfg: Route2SessionConfig,
    preview_bgr: np.ndarray | None,
    *,
    packet: DisplayPacket | None = None,
) -> np.ndarray:
    cw, ch = _canvas_size(cfg)
    if preview_bgr is None:
        return _placeholder_canvas(cw, ch)
    from core.display.live_compose import _ensure_display_eye_bgr

    preview_bgr = _ensure_display_eye_bgr(preview_bgr, cfg)
    vis = preview_bgr
    if cfg.camera_overlay and packet is not None:
        det = select_pose_det(packet.pose, cfg)
        vis = overlay_pose2d_on_preview(
            preview_bgr,
            det,
            kpts3d=packet.kpts3d or getattr(packet.pose, "kpts3d", None),
            conf=cfg.conf,
        )
    return _fit_preview(vis, cw, ch)


def poll_quit_key(cfg: Route2SessionConfig) -> bool:
    """无新帧时泵一下事件循环，避免窗口假死；返回 True 表示用户按 q。"""
    if cfg.no_display:
        return False
    key = cv2.waitKey(1) & 0xFF
    return key == ord("q")


def show_waiting_status(cfg: Route2SessionConfig, message: str) -> bool:
    """prod 无帧时保持窗口活跃并提示状态；返回 True 表示用户按 q。"""
    if cfg.no_display:
        return False
    cw, ch = resolve_display_size(cfg)
    canvas = np.full((ch, cw, 3), (32, 32, 36), np.uint8)
    for i, line in enumerate(message.split("\n")[:6]):
        cv2.putText(
            canvas,
            line,
            (16, 36 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 200, 220),
            1,
            cv2.LINE_AA,
        )
    return show_frame(cfg, canvas, message.replace("\n", "  "))


def show_frame(
    cfg: Route2SessionConfig,
    composite: np.ndarray,
    status_line: str,
) -> bool:
    """返回 True 表示用户按 q 退出。"""
    ch, cw = composite.shape[:2]
    if _embedded_preview_mode:
        # kiosk 洞内：无状态条，少一次缓冲拷贝
        _ensure_window(cw, ch)
        cv2.imshow(DISPLAY_WINDOW, composite)
    else:
        vis = _ensure_vis_buffer(cw, ch)
        np.copyto(vis[:ch, :cw], composite)
        bar = vis[ch:, :cw]
        bar[:] = _status_bar_fill
        cv2.putText(
            bar,
            status_line,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
        _ensure_window(cw, vis.shape[0])
        cv2.imshow(DISPLAY_WINDOW, vis)
    global _display_first_frame_shown
    delay = 30 if not _display_first_frame_shown else 1
    key = cv2.waitKey(delay) & 0xFF
    _display_first_frame_shown = True
    return key == ord("q")
