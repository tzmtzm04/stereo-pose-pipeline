"""伸展运动：肩角峰谷切段（4 动作类 × 每类 4 子动作）。"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from core.offline.joint_coord_series import JointAngleSeries

# 子动作 1–4 色（BGR）
STRETCH_ACTION_COLORS: tuple[tuple[int, int, int], ...] = (
    (180, 120, 40),   # 1
    (80, 180, 60),    # 2
    (60, 120, 220),   # 3
    (180, 80, 180),   # 4
)

STRETCH_TIMELINE_H = 78
_TIMELINE_MARGIN = 16
_CLASS_ROWS = 4
_ROW_H = 14
_ROW_GAP = 2
_LABEL_W = 28


@dataclass(frozen=True)
class StretchActionSegment:
    class_id: int
    action_id: int
    start_index: int
    end_index: int
    end_capture_seq: int
    end_angle_deg: float
    extremum_kind: str  # "peak" | "valley"


@dataclass(frozen=True)
class StretchActionClass:
    class_id: int
    segments: tuple[StretchActionSegment, ...]

    @property
    def start_index(self) -> int:
        return int(self.segments[0].start_index) if self.segments else 0

    @property
    def end_index(self) -> int:
        return int(self.segments[-1].end_index) if self.segments else 0


@dataclass(frozen=True)
class StretchShoulderSegmentation:
    classes: tuple[StretchActionClass, ...]
    segments: tuple[StretchActionSegment, ...]
    signal: np.ndarray
    notes: tuple[str, ...] = ()

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def _shoulder_envelope(series: JointAngleSeries) -> np.ndarray:
    sh = series.groups["shoulder"]
    return np.fmax(sh.left_deg, sh.right_deg)


def _local_extrema(arr: np.ndarray) -> list[tuple[str, int, float]]:
    out: list[tuple[str, int, float]] = []
    n = len(arr)
    for i in range(1, n - 1):
        v = float(arr[i])
        if not np.isfinite(v):
            continue
        if v >= float(arr[i - 1]) and v >= float(arr[i + 1]):
            out.append(("peak", i, v))
        elif v <= float(arr[i - 1]) and v <= float(arr[i + 1]):
            out.append(("valley", i, v))
    return out


def _alternating_extrema(
    extrema: list[tuple[str, int, float]],
    *,
    same_kind_min_dist: int = 25,
) -> list[tuple[str, int, float]]:
    """保留交替极值；同型过近时留更极端者。"""
    if not extrema:
        return []
    out: list[tuple[str, int, float]] = [extrema[0]]
    for kind, idx, val in extrema[1:]:
        pk, pi, pv = out[-1]
        if kind == pk:
            if idx - pi < int(same_kind_min_dist):
                if kind == "peak" and val > pv:
                    out[-1] = (kind, idx, val)
                elif kind == "valley" and val < pv:
                    out[-1] = (kind, idx, val)
            else:
                out.append((kind, idx, val))
        else:
            out.append((kind, idx, val))
    return out


def _find_pv_start(alt: list[tuple[str, int, float]]) -> int:
    for i in range(max(0, len(alt) - 3)):
        kinds = [alt[i + j][0] for j in range(4)]
        if kinds == ["peak", "valley", "peak", "valley"]:
            return i
    return 0


def _pick_action_classes(
    alt: list[tuple[str, int, float]],
    *,
    n_classes: int,
    n_actions: int,
) -> list[list[tuple[str, int, float]]]:
    start = _find_pv_start(alt)
    classes: list[list[tuple[str, int, float]]] = []
    need = int(n_actions)
    for ci in range(int(n_classes)):
        i0 = start + ci * need
        chunk = alt[i0 : i0 + need]
        if len(chunk) < need:
            break
        kinds = [c[0] for c in chunk]
        if kinds != ["peak", "valley", "peak", "valley"]:
            break
        classes.append(chunk)
    return classes


def _build_segments_for_class(
    class_id: int,
    picked: list[tuple[str, int, float]],
    *,
    series: JointAngleSeries,
    prev_end: int,
) -> tuple[StretchActionClass, int]:
    segments: list[StretchActionSegment] = []
    prev = max(0, int(prev_end))
    for ai, (kind, ti, ang) in enumerate(picked, start=1):
        segments.append(
            StretchActionSegment(
                class_id=int(class_id),
                action_id=int(ai),
                start_index=prev,
                end_index=int(ti),
                end_capture_seq=int(series.capture_seqs[ti]),
                end_angle_deg=float(ang),
                extremum_kind=str(kind),
            )
        )
        prev = int(ti) + 1
    cls = StretchActionClass(class_id=int(class_id), segments=tuple(segments))
    return cls, prev


def detect_stretch_shoulder_segments(
    series: JointAngleSeries,
    *,
    same_kind_min_dist: int = 25,
    n_classes: int = 4,
    n_actions: int = 4,
) -> StretchShoulderSegmentation:
    signal = _shoulder_envelope(series)
    raw = _local_extrema(signal)
    alt = _alternating_extrema(raw, same_kind_min_dist=same_kind_min_dist)
    class_chunks = _pick_action_classes(alt, n_classes=n_classes, n_actions=n_actions)

    if len(class_chunks) < n_classes:
        notes = (
            f"WARN: 动作类不足 {len(class_chunks)}/{n_classes} "
            f"(alt_extrema={len(alt)})",
        )
    else:
        notes = (
            f"shoulder envelope: {n_classes} classes × {n_actions} actions (P-V-P-V each)",
            f"smooth={series.smooth_level} same_kind_min_dist={same_kind_min_dist}",
        )

    classes: list[StretchActionClass] = []
    flat: list[StretchActionSegment] = []
    prev_end = 0
    for ci, chunk in enumerate(class_chunks, start=1):
        cls, prev_end = _build_segments_for_class(
            ci, chunk, series=series, prev_end=prev_end,
        )
        classes.append(cls)
        flat.extend(cls.segments)

    return StretchShoulderSegmentation(
        classes=tuple(classes),
        segments=tuple(flat),
        signal=signal,
        notes=notes,
    )


def draw_stretch_action_timeline(
    canvas: np.ndarray,
    *,
    y0: int,
    width: int,
    n: int,
    seg: StretchShoulderSegmentation,
    timeline_index: int = 0,
) -> None:
    """四行动作类；每行内 1–4 子动作上色并标数字。"""
    if n <= 0 or not seg.classes:
        return

    x0 = _TIMELINE_MARGIN + _LABEL_W
    x1 = width - _TIMELINE_MARGIN
    track_w = max(1, x1 - x0)
    block_y0 = y0 + 2

    cv2.rectangle(
        canvas,
        (_TIMELINE_MARGIN, block_y0),
        (width - _TIMELINE_MARGIN, block_y0 + _CLASS_ROWS * (_ROW_H + _ROW_GAP)),
        (32, 32, 38),
        -1,
    )

    for row, cls in enumerate(seg.classes):
        ry0 = block_y0 + row * (_ROW_H + _ROW_GAP)
        ry1 = ry0 + _ROW_H
        cv2.rectangle(canvas, (x0, ry0), (x1, ry1), (40, 40, 48), -1)
        cv2.rectangle(canvas, (x0, ry0), (x1, ry1), (90, 90, 100), 1)
        cv2.putText(
            canvas,
            f"C{cls.class_id}",
            (_TIMELINE_MARGIN + 2, ry1 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (170, 170, 180),
            1,
            cv2.LINE_AA,
        )

        # 类外区间置灰
        c0 = max(0, cls.start_index)
        c1 = min(n - 1, cls.end_index)
        gx0 = x0 + int((c0 / max(n - 1, 1)) * track_w)
        gx1 = x0 + int((c1 / max(n - 1, 1)) * track_w) + 1
        if c0 > 0:
            cv2.rectangle(canvas, (x0, ry0 + 1), (gx0, ry1 - 1), (45, 45, 52), -1)
        if c1 < n - 1:
            fx_tail = x0 + int((c1 / max(n - 1, 1)) * track_w) + 1
            cv2.rectangle(canvas, (fx_tail, ry0 + 1), (x1, ry1 - 1), (45, 45, 52), -1)

        for s in cls.segments:
            i0 = max(0, int(s.start_index))
            i1 = min(n - 1, int(s.end_index))
            fx0 = x0 + int((i0 / max(n - 1, 1)) * track_w)
            fx1 = x0 + int((i1 / max(n - 1, 1)) * track_w) + 1
            color = STRETCH_ACTION_COLORS[(s.action_id - 1) % 4]
            cv2.rectangle(canvas, (fx0, ry0 + 1), (max(fx0 + 1, fx1), ry1 - 1), color, -1)
            mid_x = (fx0 + fx1) // 2
            mid_y = ry1 - 4
            label = str(s.action_id)
            cv2.putText(canvas, label, (mid_x - 4, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 245, 250), 2, cv2.LINE_AA)
            cv2.putText(canvas, label, (mid_x - 4, mid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 24), 1, cv2.LINE_AA)
            tx = x0 + int((i1 / max(n - 1, 1)) * track_w)
            cv2.line(canvas, (tx, ry0), (tx, ry1), (240, 240, 240), 1, cv2.LINE_AA)

    if 0 <= timeline_index < n:
        px = x0 + int((timeline_index / max(n - 1, 1)) * track_w)
        y_top = block_y0
        y_bot = block_y0 + _CLASS_ROWS * (_ROW_H + _ROW_GAP) - _ROW_GAP
        cv2.line(canvas, (px, y_top), (px, y_bot), (80, 255, 120), 1, cv2.LINE_AA)

    scrub_y = y0 + STRETCH_TIMELINE_H - 16
    scrub_h = 8
    cv2.rectangle(
        canvas,
        (_TIMELINE_MARGIN, scrub_y - scrub_h // 2),
        (width - _TIMELINE_MARGIN, scrub_y + scrub_h // 2),
        (52, 52, 62),
        -1,
    )
    cv2.rectangle(
        canvas,
        (_TIMELINE_MARGIN, scrub_y - scrub_h // 2),
        (width - _TIMELINE_MARGIN, scrub_y + scrub_h // 2),
        (80, 80, 92),
        1,
    )
    if n > 1:
        frac = timeline_index / (n - 1)
        px = int(_TIMELINE_MARGIN + frac * (width - 2 * _TIMELINE_MARGIN))
        cv2.line(
            canvas,
            (_TIMELINE_MARGIN, scrub_y),
            (px, scrub_y),
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        cv2.circle(canvas, (px, scrub_y), 6, (0, 210, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, scrub_y), 6, (240, 240, 240), 1, cv2.LINE_AA)

    cv2.putText(
        canvas,
        "4 classes × actions 1-4 (shoulder P-V-P-V)",
        (_TIMELINE_MARGIN, y0 + STRETCH_TIMELINE_H - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (130, 130, 150),
        1,
        cv2.LINE_AA,
    )
