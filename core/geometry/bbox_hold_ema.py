"""YOLO 框：放大门控 hold + EMA breakout（ViTPose crop 稳定）。"""

from __future__ import annotations

from dataclasses import dataclass

from core.geometry.bbox_ema import BboxEmaConfig, BboxEmaFilter, BoxXYXY
from core.offline.two_stage.crop_utils import expand_bbox_xyxy


@dataclass(frozen=True)
class BboxHoldEmaConfig:
    """raw ⊆ expand(core, pad) 时冻结 gate；否则 core←EMA(raw) 并输出 expand(core, pad)。"""

    enabled: bool = False
    ema: BboxEmaConfig | None = None

    def ema_cfg(self) -> BboxEmaConfig:
        return self.ema if self.ema is not None else BboxEmaConfig()


def bbox_xyxy_contains(outer: BoxXYXY, inner: BoxXYXY) -> bool:
    """inner 完全落在 outer 内（含边界）。"""
    ox1, oy1, ox2, oy2 = (int(outer[0]), int(outer[1]), int(outer[2]), int(outer[3]))
    ix1, iy1, ix2, iy2 = (int(inner[0]), int(inner[1]), int(inner[2]), int(inner[3]))
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


class BboxHoldEmaFilter:
    """单眼：EMA 每帧推进；core 仅在 breakout 更新；输出均为 expand(·, pad)（ViTPose 就绪）。"""

    def __init__(
        self,
        cfg: BboxHoldEmaConfig,
        *,
        img_w: int,
        img_h: int,
        pad_ratio: float = 0.0,
    ) -> None:
        self._cfg = cfg
        self._img_w = max(1, int(img_w))
        self._img_h = max(1, int(img_h))
        self._pad_ratio = max(0.0, float(pad_ratio))
        self._ema = BboxEmaFilter(cfg.ema_cfg(), img_w=self._img_w, img_h=self._img_h)
        self._core: BoxXYXY | None = None
        self._conf: float = 0.0

    def reset(self) -> None:
        self._ema.reset()
        self._core = None
        self._conf = 0.0

    @property
    def has_state(self) -> bool:
        return self._core is not None

    def _expanded(self, base: BoxXYXY) -> BoxXYXY:
        if self._pad_ratio <= 0.0:
            return base
        return expand_bbox_xyxy(
            base,
            self._img_w,
            self._img_h,
            pad_ratio=self._pad_ratio,
        )

    def update(
        self,
        bbox: BoxXYXY | None,
        conf: float,
    ) -> tuple[BoxXYXY, float] | None:
        cfg = self._cfg
        if not cfg.enabled:
            return self._ema.update(bbox, conf)

        if bbox is None:
            ema_cfg = cfg.ema_cfg()
            if ema_cfg.hold_on_miss and self._core is not None:
                self._ema.update(None, 0.0)
                return self._expanded(self._core), self._conf
            self._ema.update(None, 0.0)
            return None

        ema_out = self._ema.update(bbox, conf)
        if ema_out is None:
            return None
        ema_box, ema_conf = ema_out
        self._conf = float(ema_conf)

        if self._core is None:
            self._core = ema_box
            return self._expanded(ema_box), ema_conf

        gate = self._expanded(self._core)
        if bbox_xyxy_contains(gate, bbox):
            return gate, ema_conf

        self._core = ema_box
        return self._expanded(ema_box), ema_conf
