"""SBS NVMM NV12 fd → VIC ROI 裁切（可选同次 Rotate90）→ 左/右眼 BGR。

左右眼串行：左眼完整 VIC→BGR 后再做右眼（共用全局 VIC 锁）。
竖装时 dst 表面为互换尺寸（1920×1080 ROI → 1080×1920），YOLO 直接吃竖图。
"""

from __future__ import annotations

from core.geometry.eye_mount_rotate import (
    EyeMountRotate,
    parse_eye_mount_rotate,
    rotated_eye_hw,
)
from platform.jetson.nv12_frame import Nv12Frame
from platform.jetson.nvbuf_vic_scale import (
    JetsonNvBufVicScaler,
    NvBufSurfTransform_Rotate90,
    NvBufSurfTransform_Rotate270,
    nvbuf_vic_scale_available,
)


def _vic_flip_for_rotate(mode: EyeMountRotate) -> int | None:
    # NVIDIA NvBufSurfTransform_Rotate90 与本仓库 CPU cw90 方向相反：
    #   CPU cw90  ≡ VIC Rotate270
    #   CPU ccw90 ≡ VIC Rotate90
    # （已用同帧 land→rotate MAD 对照确认）
    if mode == "cw90":
        return NvBufSurfTransform_Rotate270
    if mode == "ccw90":
        return NvBufSurfTransform_Rotate90
    return None


class SbsNv12VicSplitter:
    """3840 SBS fd 经 VIC 串行裁出单眼 BGR（可选 crop+rotate 一次完成）。"""

    def __init__(
        self,
        eye_w: int,
        eye_h: int,
        eye_rotate: EyeMountRotate | str = "none",
    ) -> None:
        if not nvbuf_vic_scale_available():
            raise RuntimeError("VIC 不可用")
        self._src_eye_w = int(eye_w)
        self._src_eye_h = int(eye_h)
        self._rotate: EyeMountRotate = parse_eye_mount_rotate(
            eye_rotate if isinstance(eye_rotate, str) else str(eye_rotate)
        )
        out_h, out_w = rotated_eye_hw(self._src_eye_h, self._src_eye_w, self._rotate)
        self._eye_w = int(out_w)
        self._eye_h = int(out_h)
        flip = _vic_flip_for_rotate(self._rotate)
        self._left = JetsonNvBufVicScaler(
            self._eye_w, self._eye_h, transform_flip=flip
        )
        self._right = JetsonNvBufVicScaler(
            self._eye_w, self._eye_h, transform_flip=flip
        )

    @property
    def out_eye_w(self) -> int:
        return self._eye_w

    @property
    def out_eye_h(self) -> int:
        return self._eye_h

    def split_eye_bgr(self, sbs: Nv12Frame) -> tuple[object, object]:
        if sbs.dmabuf_fd is None:
            raise ValueError("Nv12Frame 需要 dmabuf_fd")
        fd = int(sbs.dmabuf_fd)
        full_w, full_h = int(sbs.width), int(sbs.height)
        ew = self._src_eye_w
        # 左眼完整（VIC crop[+rotate] → BGR）后再右眼
        self._left.roi_rescale_fd(fd, 0, 0, ew, full_h)
        left_bgr = self._left.snapshot_bgr_cpu()
        self._right.roi_rescale_fd(fd, ew, 0, full_w, full_h)
        right_bgr = self._right.snapshot_bgr_cpu()
        return left_bgr, right_bgr

    def close(self) -> None:
        self._left.release()
        self._right.release()
