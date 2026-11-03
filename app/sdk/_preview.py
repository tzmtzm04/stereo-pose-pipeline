"""嵌套预览几何（SDK 私有）：GL 或 OpenCV CPU 窗。"""

from __future__ import annotations

from app.sdk.types import ScreenRect

_DEFAULT_SRC_ASPECT = 16.0 / 9.0


def cover_crop_uv(
    *,
    win_w: int,
    win_h: int,
    src_aspect: float = _DEFAULT_SRC_ASPECT,
) -> tuple[float, float, float, float]:
    if win_w <= 0 or win_h <= 0 or src_aspect <= 0:
        return 0.0, 0.0, 1.0, 1.0
    win_aspect = float(win_w) / float(win_h)
    if src_aspect > win_aspect:
        u_span = win_aspect / src_aspect
        u0 = 0.5 * (1.0 - u_span)
        return u0, 0.0, u0 + u_span, 1.0
    v_span = src_aspect / win_aspect
    v0 = 0.5 * (1.0 - v_span)
    return 0.0, v0, 1.0, v0 + v_span


def _geom_close(
    a: tuple[int, int, int, int] | None,
    b: tuple[int, int, int, int],
    *,
    tol: int = 4,
) -> bool:
    if a is None:
        return False
    return (
        abs(a[0] - b[0]) <= tol
        and abs(a[1] - b[1]) <= tol
        and abs(a[2] - b[2]) <= tol
        and abs(a[3] - b[3]) <= tol
    )


def align_gl_to_rect(
    gl,
    rect: ScreenRect,
    *,
    inset_px: int = 0,
    cover: bool = True,
    src_aspect: float = _DEFAULT_SRC_ASPECT,
    last_geom: tuple[int, int, int, int] | None = None,
    visible: bool = False,
    was_visible: bool = False,
    force: bool = False,
) -> tuple[int, int, int, int]:
    inset = max(0, int(inset_px))
    w = max(64, int(rect.width) - inset * 2)
    h = max(64, int(rect.height) - inset * 2)
    x = int(rect.x) + inset
    y = int(rect.y) + inset
    geom = (x, y, w, h)

    same = _geom_close(last_geom, geom)
    if same and visible == was_visible and not force:
        return geom

    crop = cover_crop_uv(win_w=w, win_h=h, src_aspect=src_aspect) if cover else None
    gl.request_embed(x, y, w, h, mapped=visible, crop_uv=crop)
    return geom


def hide_gl(gl) -> None:
    gl.request_hide()


def align_opencv_to_rect(
    rect: ScreenRect,
    *,
    inset_px: int = 0,
    last_geom: tuple[int, int, int, int] | None = None,
    visible: bool = False,
    was_visible: bool = False,
    force: bool = False,
) -> tuple[int, int, int, int]:
    from orchestration.display_ui import align_display_window

    inset = max(0, int(inset_px))
    w = max(64, int(rect.width) - inset * 2)
    h = max(64, int(rect.height) - inset * 2)
    x = int(rect.x) + inset
    y = int(rect.y) + inset
    geom = (x, y, w, h)

    same = _geom_close(last_geom, geom)
    if same and visible == was_visible and not force:
        return geom

    return align_display_window(x, y, w, h, mapped=visible)


def hide_opencv() -> None:
    from orchestration.display_ui import hide_display_window

    hide_display_window()


def opencv_window_ready() -> bool:
    from orchestration.display_ui import display_window_is_open

    return display_window_is_open()
