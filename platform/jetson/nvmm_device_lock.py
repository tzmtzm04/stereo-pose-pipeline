"""Serialize session enc pool blit vs TRT CUDA（仅 --session-rec-mode pool 路径）。"""

from __future__ import annotations

import threading

# RLock：infer 双眼 TRT 线程池可能嵌套回调
NVENC_INFER_LOCK = threading.RLock()
