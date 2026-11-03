"""YOLO 人体框时序 EMA 平滑（单人单眼，xyxy 四边独立）。"""

from __future__ import annotations

import math
from dataclasses import dataclass

BoxXYXY = tuple[int, int, int, int]


@dataclass(frozen=True)
class BboxEmaConfig:
    """轻量 EMA：高 alpha、不限制 max_step，只消 p95 级小跳，快动跟手。"""
    enabled: bool = True
    alpha: float = 0.55
    alpha_size: float = 0.40
    teleport_px: float = 120.0
    hold_on_miss: bool = True
    max_center_step_px: float = 0.0
    max_size_step_px: float = 0.0


def _clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    img_w: int,
    img_h: int,
) -> BoxXYXY:
    iw = max(1, int(img_w))
    ih = max(1, int(img_h))
    x1i = max(0, min(iw - 2, int(round(x1))))
    y1i = max(0, min(ih - 2, int(round(y1))))
    x2i = max(x1i + 1, min(iw - 1, int(round(x2))))
    y2i = max(y1i + 1, min(ih - 1, int(round(y2))))
    return (x1i, y1i, x2i, y2i)


class BboxEmaFilter:
    """对 x1/y1/x2/y2 分别 EMA；中心跳变超过 teleport_px 时重置。"""

    def __init__(
        self,
        cfg: BboxEmaConfig,
        *,
        img_w: int,
        img_h: int,
    ) -> None:
        self._cfg = cfg
        self._img_w = max(1, int(img_w))
        self._img_h = max(1, int(img_h))
        self._x1 = 0.0
        self._y1 = 0.0
        self._x2 = 0.0
        self._y2 = 0.0
        self._conf = 0.0
        self._has_state = False

    def reset(self) -> None:
        self._has_state = False

    @property
    def has_state(self) -> bool:
        return bool(self._has_state)

    def _blend(self, alpha: float, cur: float, prev: float) -> float:
        a = float(max(0.0, min(1.0, alpha)))
        return a * float(cur) + (1.0 - a) * float(prev)

    def _clamp_step(self, target: float, prev: float, max_step: float) -> float:
        step = float(max_step)
        if step <= 0.0:
            return float(target)
        delta = float(target) - float(prev)
        if delta > step:
            return float(prev) + step
        if delta < -step:
            return float(prev) - step
        return float(target)

    def _center(self) -> tuple[float, float]:
        return (self._x1 + self._x2) * 0.5, (self._y1 + self._y2) * 0.5

    def _output(self, conf: float) -> tuple[BoxXYXY, float]:
        return (
            _clip_xyxy(
                self._x1,
                self._y1,
                self._x2,
                self._y2,
                img_w=self._img_w,
                img_h=self._img_h,
            ),
            float(conf),
        )

    def update(
        self,
        bbox: BoxXYXY | None,
        conf: float,
    ) -> tuple[BoxXYXY, float] | None:
        cfg = self._cfg
        if not cfg.enabled:
            if bbox is None:
                return None
            x1, y1, x2, y2 = (float(v) for v in bbox[:4])
            return _clip_xyxy(x1, y1, x2, y2, img_w=self._img_w, img_h=self._img_h), float(
                conf
            )

        if bbox is None:
            if cfg.hold_on_miss and self._has_state:
                return self._output(self._conf)
            return None

        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        conf_f = float(conf)
        if not self._has_state:
            self._set_state(x1, y1, x2, y2, conf_f)
            return self._output(conf_f)

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        pcx, pcy = self._center()
        jump = math.hypot(cx - pcx, cy - pcy)
        if jump > float(cfg.teleport_px):
            self._set_state(x1, y1, x2, y2, conf_f)
            return self._output(conf_f)

        alpha_xy = float(cfg.alpha)
        alpha_edge = float(cfg.alpha_size)
        x1_s = self._clamp_step(
            self._blend(alpha_xy, x1, self._x1),
            self._x1,
            cfg.max_center_step_px,
        )
        x2_s = self._clamp_step(
            self._blend(alpha_xy, x2, self._x2),
            self._x2,
            cfg.max_center_step_px,
        )
        y1_s = self._clamp_step(
            self._blend(alpha_edge, y1, self._y1),
            self._y1,
            cfg.max_size_step_px,
        )
        y2_s = self._clamp_step(
            self._blend(alpha_edge, y2, self._y2),
            self._y2,
            cfg.max_size_step_px,
        )
        if x2_s <= x1_s + 1.0:
            x2_s = x1_s + 1.0
        if y2_s <= y1_s + 1.0:
            y2_s = y1_s + 1.0
        self._set_state(x1_s, y1_s, x2_s, y2_s, conf_f)
        return self._output(conf_f)

    def _set_state(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        conf: float,
    ) -> None:
        self._x1 = float(x1)
        self._y1 = float(y1)
        self._x2 = float(x2)
        self._y2 = float(y2)
        self._conf = float(conf)
        self._has_state = True
