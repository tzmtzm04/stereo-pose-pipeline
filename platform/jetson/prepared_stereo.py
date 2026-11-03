"""采集线程 VIC 预处理产物：双眼 letterbox 640 NV12 → EGL-CUDA planes（infer 仅 TRT）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from platform.jetson.letterbox_util import LetterboxGeometry
from platform.jetson.nvbuf_egl_cuda import Nv12CudaPitchPlanes


@dataclass
class PreparedEyeTRT:
    """单眼 TRT 输入（letterbox 后 640×640 NV12 CUDA pitch）。"""

    ow: int
    oh: int
    geo: LetterboxGeometry
    planes: Nv12CudaPitchPlanes
    vic_ms: float = 0.0
    egl_ms: float = 0.0


@dataclass
class PreparedStereoFrame:
    """capture 线程完成 split + letterbox 后交给 infer 线程。"""

    left: PreparedEyeTRT
    right: PreparedEyeTRT
    capture_seq: int = -1
    pull_ms: float = 0.0
    vic_prep_ms: float = 0.0
    split_ms: float = 0.0
    letterbox_ms: float = 0.0
    preview_bgr: np.ndarray | None = None
    preview_ms: float = 0.0
    _slot_keepalive: Any = field(default=None, repr=False)

    @property
    def capture_ms(self) -> float:
        return self.pull_ms + self.vic_prep_ms

    def release(self) -> None:
        """释放 NVMM 槽位；preview_bgr 为 CPU 拷贝，由 infer/display 持有。"""
        self._slot_keepalive = None
