"""单人 YOLO bbox Kalman 跟踪（TRT det 后处理，支持 det_step 跳帧预测）。"""

from __future__ import annotations

import math
from dataclasses import dataclass

BoxXYXY = tuple[int, int, int, int]


@dataclass(frozen=True)
class BboxTrackerConfig:
    enabled: bool = False
    det_step: int = 2
    constant_size: bool = True
    process_noise: float = 2.0
    measure_noise: float = 6.0
    max_miss: int = 8
    teleport_px: float = 180.0
    fps: float = 60.0


def _xyxy_to_cxcywh(box: BoxXYXY) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = (float(box[0]), float(box[1]), float(box[2]), float(box[3]))
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5, w, h


def _cxcywh_to_xyxy(
    cx: float,
    cy: float,
    w: float,
    h: float,
    *,
    img_w: int,
    img_h: int,
) -> BoxXYXY:
    iw = max(1, int(img_w))
    ih = max(1, int(img_h))
    half_w = max(0.5, float(w) * 0.5)
    half_h = max(0.5, float(h) * 0.5)
    x1 = int(max(0, min(iw - 2, round(cx - half_w))))
    y1 = int(max(0, min(ih - 2, round(cy - half_h))))
    x2 = int(max(x1 + 1, min(iw - 1, round(cx + half_w))))
    y2 = int(max(y1 + 1, min(ih - 1, round(cy + half_h))))
    return x1, y1, x2, y2


class _Kalman1D:
    """常速度模型 [pos, vel]。"""

    def __init__(
        self,
        *,
        process_noise: float,
        measure_noise: float,
    ) -> None:
        self._q = float(max(1e-6, process_noise))
        self._r = float(max(1e-6, measure_noise))
        self._pos = 0.0
        self._vel = 0.0
        self._p00 = 1.0
        self._p01 = 0.0
        self._p11 = 1.0
        self._initialized = False

    def reset(self) -> None:
        self._initialized = False
        self._pos = 0.0
        self._vel = 0.0
        self._p00 = 1.0
        self._p01 = 0.0
        self._p11 = 1.0

    @property
    def initialized(self) -> bool:
        return bool(self._initialized)

    @property
    def pos(self) -> float:
        return float(self._pos)

    def predict(self, dt: float) -> float:
        dt_f = float(max(1e-6, dt))
        if not self._initialized:
            return float(self._pos)
        self._pos += self._vel * dt_f
        self._p00 += dt_f * (dt_f * self._p11 + 2.0 * self._p01) + self._q * dt_f * dt_f
        self._p01 += dt_f * self._p11
        self._p11 += self._q
        return float(self._pos)

    def update(self, z: float, *, dt: float = 1.0) -> float:
        if not self._initialized:
            self._pos = float(z)
            self._vel = 0.0
            self._p00 = self._r
            self._p01 = 0.0
            self._p11 = self._q
            self._initialized = True
            return float(self._pos)
        self.predict(dt)
        s = self._p00 + self._r
        k0 = self._p00 / s
        k1 = self._p01 / s
        y = float(z) - self._pos
        self._pos += k0 * y
        self._vel += k1 * y
        p00_old = self._p00
        p01_old = self._p01
        self._p00 = (1.0 - k0) * p00_old
        self._p01 = (1.0 - k0) * p01_old
        self._p11 = self._p11 - k1 * p01_old
        return float(self._pos)


class SinglePersonBboxTracker:
    """单眼单人 bbox：Kalman(cx,cy[,w,h]) + det_step 跳帧 predict。"""

    def __init__(
        self,
        cfg: BboxTrackerConfig,
        *,
        img_w: int,
        img_h: int,
    ) -> None:
        self._cfg = cfg
        self._img_w = max(1, int(img_w))
        self._img_h = max(1, int(img_h))
        q = float(cfg.process_noise)
        r = float(cfg.measure_noise)
        self._cx = _Kalman1D(process_noise=q, measure_noise=r)
        self._cy = _Kalman1D(process_noise=q, measure_noise=r)
        self._w = _Kalman1D(process_noise=q * 0.25, measure_noise=r * 2.0)
        self._h = _Kalman1D(process_noise=q * 0.25, measure_noise=r * 2.0)
        self._conf = 0.0
        self._locked_w = 0.0
        self._locked_h = 0.0
        self._miss = 0

    def reset(self) -> None:
        self._cx.reset()
        self._cy.reset()
        self._w.reset()
        self._h.reset()
        self._conf = 0.0
        self._locked_w = 0.0
        self._locked_h = 0.0
        self._miss = 0

    def _dt(self) -> float:
        fps = float(max(1.0, self._cfg.fps))
        step = float(max(1, int(self._cfg.det_step)))
        return step / fps

    def _has_state(self) -> bool:
        return self._cx.initialized

    def _teleport(self, box: BoxXYXY) -> bool:
        if not self._has_state():
            return False
        cx, cy, _, _ = _xyxy_to_cxcywh(box)
        jump = math.hypot(cx - self._cx.pos, cy - self._cy.pos)
        return jump > float(self._cfg.teleport_px)

    def _output_box(self) -> BoxXYXY:
        if self._cfg.constant_size and self._locked_w > 1.0 and self._locked_h > 1.0:
            w = self._locked_w
            h = self._locked_h
        else:
            w = max(1.0, self._w.pos if self._w.initialized else self._locked_w)
            h = max(1.0, self._h.pos if self._h.initialized else self._locked_h)
        return _cxcywh_to_xyxy(
            self._cx.pos,
            self._cy.pos,
            w,
            h,
            img_w=self._img_w,
            img_h=self._img_h,
        )

    def observe(
        self,
        bbox: BoxXYXY | None,
        conf: float,
    ) -> tuple[BoxXYXY, float] | None:
        """det 帧：用 YOLO 观测更新（bbox=None 表示 det 未检出）。"""
        if bbox is not None and self._teleport(bbox):
            self.reset()
        dt = self._dt()
        if bbox is None:
            self._miss += 1
            if not self._has_state() or self._miss > int(self._cfg.max_miss):
                return None
            self._cx.predict(dt)
            self._cy.predict(dt)
            if not self._cfg.constant_size:
                self._w.predict(dt)
                self._h.predict(dt)
            return self._output_box(), float(self._conf)

        cx, cy, w, h = _xyxy_to_cxcywh(bbox)
        self._cx.update(cx, dt=dt)
        self._cy.update(cy, dt=dt)
        if self._cfg.constant_size:
            if self._locked_w <= 1.0 or self._locked_h <= 1.0:
                self._locked_w = float(w)
                self._locked_h = float(h)
        else:
            self._w.update(w, dt=dt)
            self._h.update(h, dt=dt)
        self._conf = float(conf)
        self._miss = 0
        return self._output_box(), float(self._conf)

    def predict_only(self) -> tuple[BoxXYXY, float] | None:
        """非 det 帧：仅 Kalman 外推。"""
        if not self._has_state():
            return None
        dt = self._dt()
        self._cx.predict(dt)
        self._cy.predict(dt)
        if not self._cfg.constant_size:
            self._w.predict(dt)
            self._h.predict(dt)
        self._miss += 1
        if self._miss > int(self._cfg.max_miss):
            return None
        return self._output_box(), float(self._conf)
