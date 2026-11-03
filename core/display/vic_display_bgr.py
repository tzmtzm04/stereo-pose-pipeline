"""display 原画：与 infer prep 同 VIC ROI 裁切 → BGR（策略 D 仍独立 pull）。"""

from __future__ import annotations

import numpy as np

from core.display.display_eye import normalize_display_eye
from core.geometry.eye_mount_rotate import (
    EyeMountRotate,
    parse_eye_mount_rotate,
)
from core.offline.sbs_nv12_vic_split import SbsNv12VicSplitter
from platform.jetson.nv12_frame import Nv12Frame


class DisplayVicEyeBgr:
    """全幅 SBS NV12 fd → 单眼 BGR（同 SbsNv12VicSplitter / infer prep）。"""

    def __init__(
        self,
        *,
        eye_w: int,
        eye_h: int,
        eye_rotate: EyeMountRotate | str = "none",
    ) -> None:
        self._eye_w = int(eye_w)
        self._eye_h = int(eye_h)
        self._eye_rotate: EyeMountRotate = parse_eye_mount_rotate(
            eye_rotate if isinstance(eye_rotate, str) else str(eye_rotate)
        )
        # 旋转在 VIC crop 同次完成；勿再 CPU 二次旋转
        self._splitter = SbsNv12VicSplitter(
            self._eye_w, self._eye_h, eye_rotate=self._eye_rotate
        )

    def eye_bgr_from_sbs(self, sbs: Nv12Frame, *, display_eye: str) -> np.ndarray:
        if sbs.dmabuf_fd is None:
            raise ValueError("display VIC 解码需要 NV12 dmabuf_fd")
        left_bgr, right_bgr = self._splitter.split_eye_bgr(sbs)
        raw = (
            right_bgr
            if normalize_display_eye(display_eye) == "right"
            else left_bgr
        )
        return np.ascontiguousarray(raw)

    def close(self) -> None:
        self._splitter.close()
