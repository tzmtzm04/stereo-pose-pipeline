"""EGL+GL 硬件预览（B 路线：NV12 纹理 + GL 2D 骨架）。"""

from __future__ import annotations

from app.route2.session_config import Route2SessionConfig
from core.display.live_compose import resolve_display_size
from orchestration.constants import DISPLAY_WINDOW
from platform.jetson.nv12_gl_display import (
    STAGING_ORIGIN_X,
    STAGING_ORIGIN_Y,
    Nv12GlDisplay,
    nv12_gl_display_available,
)


def hw_gl_display_available() -> bool:
    return nv12_gl_display_available()


def create_hw_gl_display(cfg: Route2SessionConfig) -> Nv12GlDisplay:
    w, h = resolve_display_size(cfg)
    # 嵌入预览：屏外创建+map，后台 swap 预热，绝不在 (0,0) 闪窗
    embed = bool(getattr(cfg, "embedded_preview", False))
    if embed:
        return Nv12GlDisplay(
            title=DISPLAY_WINDOW,
            width=w,
            height=h,
            start_mapped=True,
            override_redirect=True,
            origin_x=STAGING_ORIGIN_X,
            origin_y=STAGING_ORIGIN_Y,
            offscreen_stage=True,
        )
    return Nv12GlDisplay(
        title=DISPLAY_WINDOW,
        width=w,
        height=h,
        start_mapped=True,
        override_redirect=False,
    )
