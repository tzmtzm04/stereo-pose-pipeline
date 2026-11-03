"""vic_bgr 两阶段流水线：prep 线程 VIC→BGR→letterbox ∥ infer 线程 TRT+post（默认开）。"""

from __future__ import annotations

from app.route2.session_config import Route2SessionConfig


def vic_bgr_prep_pipeline_enabled(cfg: Route2SessionConfig) -> bool:
    """VIC→BGR→letterbox prep 线程 + prep_q + infer 取最新。"""
    return (
        bool(cfg.infer_pipeline_overlap)
        and str(getattr(cfg, "infer_prep", "vic_bgr")).strip().lower() == "vic_bgr"
    )
