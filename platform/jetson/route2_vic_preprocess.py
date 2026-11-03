"""路线 2 VIC：NVMM fd → Ultralytics LetterBox 640×640（一步 VIC）。"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .letterbox_util import LetterboxGeometry, ultralytics_letterbox_geometry
from .nvbuf_vic_scale import JetsonNvBufVicScaler, nvbuf_vic_scale_available

logger = logging.getLogger(__name__)

DEFAULT_ROUTE2_INPUT = (640, 640)  # W, H
LETTERBOX_GRAY = 114


@dataclass
class VicLetterboxProfile:
    """LetterBox VIC 分段耗时（ms）；一步 letterbox 时 place_ms=0。"""

    unpad_ms: float = 0.0
    place_ms: float = 0.0
    egl_bind_ms: float = 0.0

    @property
    def vic_total_ms(self) -> float:
        return self.unpad_ms + self.place_ms + self.egl_bind_ms


def route2_nvbuf_vic_available() -> bool:
    return nvbuf_vic_scale_available()


class Route2VicPreprocessor:
    """
    NVMM NV12 eye fd → 一步 VIC LetterBox（对齐 Ultralytics）：

    1920×1080 整幅 → 缩放到 640×640 灰底上的 unpad 区域（如 640×360 @ 居中）。
    """

    def __init__(
        self,
        det_size: tuple[int, int] = DEFAULT_ROUTE2_INPUT,
    ) -> None:
        if not route2_nvbuf_vic_available():
            raise RuntimeError("NvBuf VIC 不可用")
        self.det_w, self.det_h = int(det_size[0]), int(det_size[1])
        self._letter_scaler: JetsonNvBufVicScaler | None = None
        self._src_w = 0
        self._src_h = 0
        self._letter_gray_filled = False
        self._last_geometry: LetterboxGeometry | None = None
        self._preview_scaler: JetsonNvBufVicScaler | None = None
        self._preview_w = 0
        self._preview_h = 0

    def warm_up(self, src_w: int, src_h: int) -> None:
        """主线程预建 VIC scaler（避免首帧初始化）。"""
        geo = ultralytics_letterbox_geometry(
            int(src_w), int(src_h), self.det_w, self.det_h
        )
        self._ensure_letter_scaler(geo)

    def _release_preview_scaler(self) -> None:
        if self._preview_scaler is not None:
            self._preview_scaler.release()
            self._preview_scaler = None
        self._preview_w = 0
        self._preview_h = 0

    def _release_letter_scaler(self) -> None:
        if self._letter_scaler is not None:
            self._letter_scaler.release()
            self._letter_scaler = None
        self._src_w = 0
        self._src_h = 0
        self._letter_gray_filled = False

    def _ensure_letter_scaler(self, geo: LetterboxGeometry) -> None:
        if (
            self._letter_scaler is not None
            and self._src_w == geo.src_w
            and self._src_h == geo.src_h
        ):
            return
        self._release_letter_scaler()
        self._src_w, self._src_h = geo.src_w, geo.src_h
        self._letter_scaler = JetsonNvBufVicScaler(
            self.det_w, self.det_h, prefill_gray114=True
        )
        self._letter_gray_filled = True
        logger.info(
            "Route2Vic letterbox (1-step): %dx%d → %dx%d pad=(%d,%d) content=%dx%d",
            geo.src_w,
            geo.src_h,
            self.det_w,
            self.det_h,
            geo.pad_left,
            geo.pad_top,
            geo.unpad_w,
            geo.unpad_h,
        )

    def _run_letterbox_vic(self, src_fd: int, w: int, h: int) -> tuple[LetterboxGeometry, VicLetterboxProfile]:
        prof = VicLetterboxProfile()
        geo = ultralytics_letterbox_geometry(w, h, self.det_w, self.det_h)
        self._ensure_letter_scaler(geo)
        assert self._letter_scaler is not None

        if not self._letter_gray_filled:
            self._letter_scaler.fill_nv12_gray114()
            self._letter_gray_filled = True

        t0 = time.perf_counter()
        self._letter_scaler.letterbox_fd(
            int(src_fd),
            int(w),
            int(h),
            dst_left=geo.pad_left,
            dst_top=geo.pad_top,
            content_w=geo.unpad_w,
            content_h=geo.unpad_h,
        )
        prof.unpad_ms = (time.perf_counter() - t0) * 1000.0
        prof.place_ms = 0.0
        self._last_geometry = geo
        return geo, prof

    def _run_letterbox_roi_vic(
        self,
        src_fd: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        eye_w: int,
        eye_h: int,
    ) -> tuple[LetterboxGeometry, VicLetterboxProfile]:
        prof = VicLetterboxProfile()
        geo = ultralytics_letterbox_geometry(
            int(eye_w), int(eye_h), self.det_w, self.det_h
        )
        self._ensure_letter_scaler(geo)
        assert self._letter_scaler is not None

        if not self._letter_gray_filled:
            self._letter_scaler.fill_nv12_gray114()
            self._letter_gray_filled = True

        t0 = time.perf_counter()
        self._letter_scaler.letterbox_roi_fd(
            int(src_fd),
            int(x1),
            int(y1),
            int(x2),
            int(y2),
            dst_left=geo.pad_left,
            dst_top=geo.pad_top,
            content_w=geo.unpad_w,
            content_h=geo.unpad_h,
        )
        prof.unpad_ms = (time.perf_counter() - t0) * 1000.0
        prof.place_ms = 0.0
        self._last_geometry = geo
        return geo, prof

    @property
    def last_geometry(self) -> LetterboxGeometry | None:
        return self._last_geometry

    def prepare_bgr_from_fd(self, src_fd: int, width: int, height: int) -> Any:
        _, _ = self._run_letterbox_vic(int(src_fd), int(width), int(height))
        assert self._letter_scaler is not None
        return self._letter_scaler.snapshot_bgr_cpu()

    def preview_bgr_from_fd(self, src_fd: int, width: int, height: int) -> Any:
        """全分辨率 VIC→BGR 预览（与推理同源 fd，避免 NV12 CPU 转换比例失真）。"""
        w, h = int(width), int(height)
        if (
            self._preview_scaler is None
            or self._preview_w != w
            or self._preview_h != h
        ):
            self._release_preview_scaler()
            self._preview_scaler = JetsonNvBufVicScaler(w, h)
            self._preview_w, self._preview_h = w, h
        assert self._preview_scaler is not None
        self._preview_scaler.rescale_fd(int(src_fd), w, h)
        return self._preview_scaler.snapshot_bgr_cpu()

    def preview_unpad_upscaled_bgr(self, dst_w: int, dst_h: int) -> Any:
        """
        从一步 letterbox 的 640 输出裁 content 区放大到原图尺寸（叠图底图）。

        须在 _run_letterbox_vic / prepare_nv12_cuda_from_fd 之后调用。
        """
        import cv2

        if self._letter_scaler is None or self._last_geometry is None:
            raise RuntimeError("letterbox 未就绪，请先跑 letterbox VIC")
        geo = self._last_geometry
        bgr = self._letter_scaler.snapshot_bgr_cpu()
        y1, x1 = geo.pad_top, geo.pad_left
        y2, x2 = y1 + geo.unpad_h, x1 + geo.unpad_w
        crop = bgr[y1:y2, x1:x2]
        dw, dh = int(dst_w), int(dst_h)
        if crop.shape[1] == dw and crop.shape[0] == dh:
            return crop
        return cv2.resize(crop, (dw, dh), interpolation=cv2.INTER_LINEAR)

    def prepare_nv12_cuda_from_fd(
        self, src_fd: int, width: int, height: int
    ) -> tuple[Any, VicLetterboxProfile]:
        """LetterBox 640 NV12 → EGL-CUDA pitch（TRT 直绑）。"""
        _, prof = self._run_letterbox_vic(int(src_fd), int(width), int(height))
        assert self._letter_scaler is not None
        t0 = time.perf_counter()
        planes, egl_prof = self._letter_scaler.nv12_pitch_cuda_planes(strict_egl=True)
        prof.egl_bind_ms = (time.perf_counter() - t0) * 1000.0
        if egl_prof.get("page_nvmm_egl_cuda_bind"):
            pass  # 标记成功；耗时以 wall clock 为准
        return planes, prof

    def prepare_nv12_cuda_from_roi_fd(
        self,
        src_fd: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        eye_w: int,
        eye_h: int,
    ) -> tuple[Any, VicLetterboxProfile]:
        """SBS ROI → LetterBox 640 → EGL-CUDA pitch（无中间 split）。"""
        _, prof = self._run_letterbox_roi_vic(
            int(src_fd), int(x1), int(y1), int(x2), int(y2), int(eye_w), int(eye_h)
        )
        assert self._letter_scaler is not None
        t0 = time.perf_counter()
        planes, egl_prof = self._letter_scaler.nv12_pitch_cuda_planes(strict_egl=True)
        prof.egl_bind_ms = (time.perf_counter() - t0) * 1000.0
        if egl_prof.get("page_nvmm_egl_cuda_bind"):
            pass
        return planes, prof

    def release(self) -> None:
        self._release_letter_scaler()
        self._release_preview_scaler()
        self._last_geometry = None
