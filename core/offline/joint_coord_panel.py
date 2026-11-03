"""左栏：肩/肘/髋/膝 四宫格关节角曲线（同关节 L/R 同图）。"""

from __future__ import annotations

import math

import cv2
import numpy as np

from core.offline.joint_coord_series import (
    JOINT_GROUPS,
    _GROUP_ANGLE_HINT,
    JointAngleSeries,
)

_LEFT_COLOR = (0, 210, 255)
_RIGHT_COLOR = (140, 140, 160)
_GROUP_TITLES = {
    "shoulder": "肩 shoulder",
    "elbow": "肘 elbow",
    "hip": "髋 hip",
    "knee": "膝 knee",
}


def _plot_y(v: float, vmin: float, vmax: float, y0: int, y1: int) -> int:
    span = max(vmax - vmin, 1e-9)
    t = (float(v) - vmin) / span
    return int(y1 - t * (y1 - y0))


def _series_limits(*arrays: np.ndarray) -> tuple[float, float]:
    chunks = [a[np.isfinite(a)] for a in arrays if a.size]
    if not any(c.size for c in chunks):
        return 0.0, 180.0
    flat = np.concatenate([c for c in chunks if c.size])
    vmin = float(np.min(flat))
    vmax = float(np.max(flat))
    pad = max(0.05 * (vmax - vmin), 3.0)
    return vmin - pad, vmax + pad


def _draw_angle_curve(
    canvas: np.ndarray,
    *,
    arr: np.ndarray,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    vmin: float,
    vmax: float,
    color: tuple[int, int, int],
    thickness: int,
) -> None:
    n = len(arr)
    if n <= 0:
        return
    pts: list[tuple[int, int]] = []
    for i in range(n):
        v = float(arr[i])
        if not math.isfinite(v):
            continue
        fx = i / max(n - 1, 1)
        px = x0 + int(fx * (x1 - x0))
        py = _plot_y(v, vmin, vmax, y0, y1)
        pts.append((px, py))
    if len(pts) >= 2:
        cv2.polylines(
            canvas,
            [np.array(pts, dtype=np.int32)],
            False,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _draw_subplot(
    canvas: np.ndarray,
    *,
    group_name: str,
    pair,
    index: int,
    x0: int,
    y0: int,
    w: int,
    h: int,
) -> None:
    margin_l, margin_r = 44, 8
    margin_t, margin_b = 24, 8
    px0 = x0 + margin_l
    px1 = x0 + w - margin_r
    py0 = y0 + margin_t
    py1 = y0 + h - margin_b
    cv2.rectangle(canvas, (px0, py0), (px1, py1), (40, 40, 48), 1)

    left_raw = pair.left_deg_raw
    right_raw = pair.right_deg_raw
    left = pair.left_deg
    right = pair.right_deg
    n = len(left)
    vmin, vmax = _series_limits(left_raw, right_raw, left, right)

    _draw_angle_curve(
        canvas,
        arr=left_raw,
        x0=px0,
        x1=px1,
        y0=py0,
        y1=py1,
        vmin=vmin,
        vmax=vmax,
        color=(70, 70, 75),
        thickness=1,
    )
    _draw_angle_curve(
        canvas,
        arr=right_raw,
        x0=px0,
        x1=px1,
        y0=py0,
        y1=py1,
        vmin=vmin,
        vmax=vmax,
        color=(55, 55, 60),
        thickness=1,
    )
    _draw_angle_curve(
        canvas,
        arr=left,
        x0=px0,
        x1=px1,
        y0=py0,
        y1=py1,
        vmin=vmin,
        vmax=vmax,
        color=_LEFT_COLOR,
        thickness=2,
    )
    _draw_angle_curve(
        canvas,
        arr=right,
        x0=px0,
        x1=px1,
        y0=py0,
        y1=py1,
        vmin=vmin,
        vmax=vmax,
        color=_RIGHT_COLOR,
        thickness=1,
    )

    if n > 0 and 0 <= index < n:
        frac = index / max(n - 1, 1)
        play_x = px0 + int(frac * (px1 - px0))
        cv2.line(canvas, (play_x, py0), (play_x, py1), (80, 255, 120), 1, cv2.LINE_AA)
        la = float(left[index]) if index < len(left) else float("nan")
        if math.isfinite(la):
            my = _plot_y(la, vmin, vmax, py0, py1)
            cv2.circle(canvas, (play_x, my), 4, _LEFT_COLOR, -1, cv2.LINE_AA)

    hint = _GROUP_ANGLE_HINT.get(group_name, "")
    title = _GROUP_TITLES.get(group_name, group_name)
    if hint:
        title += f"  ({hint})"
    cv2.putText(
        canvas,
        title,
        (px0, y0 + 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (210, 210, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (px1 - 70, py0 + 8), (px1 - 56, py0 + 8), _LEFT_COLOR, 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "L",
        (px1 - 52, py0 + 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        _LEFT_COLOR,
        1,
        cv2.LINE_AA,
    )
    cv2.line(canvas, (px1 - 70, py0 + 20), (px1 - 56, py0 + 20), _RIGHT_COLOR, 1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "R",
        (px1 - 52, py0 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        _RIGHT_COLOR,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        canvas,
        f"{vmin:.0f}",
        (x0 + 2, py1),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        (120, 120, 130),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{vmax:.0f}",
        (x0 + 2, py0 + 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        (120, 120, 130),
        1,
        cv2.LINE_AA,
    )

    if 0 <= index < n:
        la = float(left[index])
        ra = float(right[index])
        txt = f"L={la:.1f}  R={ra:.1f} deg"
        if math.isfinite(la) or math.isfinite(ra):
            cv2.putText(
                canvas,
                txt,
                (px0, py1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.28,
                (150, 180, 170),
                1,
                cv2.LINE_AA,
            )


def draw_joint_angle_panel(
    *,
    series: JointAngleSeries,
    index: int,
    width: int,
    height: int,
    smooth_tag: str = "",
) -> np.ndarray:
    canvas = np.full((height, width, 3), (24, 24, 28), np.uint8)
    if series.count <= 0:
        return canvas

    cols, rows = 2, 2
    cell_w = width // cols
    cell_h = height // rows
    for gi, (name, _, _) in enumerate(JOINT_GROUPS):
        col = gi % cols
        row = gi // cols
        cx = col * cell_w
        cy = row * cell_h
        pair = series.groups[name]
        _draw_subplot(
            canvas,
            group_name=name,
            pair=pair,
            index=index,
            x0=cx,
            y0=cy,
            w=cell_w,
            h=cell_h,
        )
        if col == 0:
            cv2.line(canvas, (cx + cell_w, cy), (cx + cell_w, cy + cell_h), (50, 50, 58), 1)
        if row == 0:
            cv2.line(canvas, (cx, cy + cell_h), (cx + cell_w, cy + cell_h), (50, 50, 58), 1)

    footer = "joint angle (deg)  gaussian smooth"
    if smooth_tag:
        footer += f"  [{smooth_tag}]"
    cv2.putText(
        canvas,
        footer,
        (8, height - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (140, 140, 160),
        1,
        cv2.LINE_AA,
    )
    return canvas


draw_joint_coord_panel = draw_joint_angle_panel
