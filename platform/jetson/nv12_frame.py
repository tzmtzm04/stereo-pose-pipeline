"""NVMM NV12 帧容器（GStreamer appsink → VIC，零拷贝 fd）。"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

_KEEPALIVE: dict[int, "NvmmMapKeepalive"] = {}
_GST_SAMPLE_KEEPALIVE: dict[int, "GstSampleKeepalive"] = {}


class GstBufferKeepalive:
    """Gst.Buffer 引用计数（mjpeg ingress 异步拷贝前持有 buffer）。"""

    __slots__ = ("buf", "_refs", "_lock")

    def __init__(self, buf: Any) -> None:
        self.buf = buf
        self._refs = 1
        self._lock = threading.Lock()
        try:
            buf.ref()
        except Exception:
            pass

    def unref(self) -> None:
        with self._lock:
            self._refs -= 1
            if self._refs <= 0 and self.buf is not None:
                try:
                    self.buf.unref()
                except Exception:
                    pass
                self.buf = None


class GstSampleKeepalive:
    """Gst.Sample 引用计数（capture / infer / enc 共享 NVMM，零拷贝）。"""

    __slots__ = ("sample", "_refs", "_lock", "_sample_id")

    def __init__(self, sample: Any) -> None:
        self.sample = sample
        self._sample_id = id(sample)
        self._refs = 1
        self._lock = threading.Lock()
        _GST_SAMPLE_KEEPALIVE[self._sample_id] = self

    def ref(self) -> "GstSampleKeepalive":
        with self._lock:
            if self.sample is None:
                raise RuntimeError("GstSampleKeepalive ref on released sample")
            self._refs += 1
        return self

    def unref(self) -> None:
        sid = None
        with self._lock:
            self._refs -= 1
            if self._refs <= 0:
                sid = self._sample_id
                self.sample = None
        if sid is not None:
            _GST_SAMPLE_KEEPALIVE.pop(sid, None)


def gst_sample_keepalive_acquire(sample: Any) -> GstSampleKeepalive:
    """同一 Gst.Sample 复用 keepalive（首持有者 refs=1）。"""
    sid = id(sample)
    existing = _GST_SAMPLE_KEEPALIVE.get(sid)
    if existing is not None:
        if existing.sample is None:
            existing = None
        else:
            return existing
    return GstSampleKeepalive(sample)


def gst_sample_keepalive_from_any(keepalive: Any) -> GstSampleKeepalive | None:
    if isinstance(keepalive, GstSampleKeepalive):
        return keepalive if keepalive.sample is not None else None
    if keepalive is not None and hasattr(keepalive, "get_buffer"):
        return gst_sample_keepalive_acquire(keepalive)
    return None


class NvmmMapKeepalive:
    """Gst sample + NvBufSurface 引用计数。"""

    __slots__ = ("sample", "surf", "_refs", "_lock")

    def __init__(self, sample: Any, surf: Any) -> None:
        self.sample = sample
        self.surf = surf
        self._refs = 1
        self._lock = threading.Lock()
        _KEEPALIVE[id(surf)] = self

    @classmethod
    def adopt(cls, keepalive: Any) -> "NvmmMapKeepalive | None":
        if keepalive is None:
            return None
        if isinstance(keepalive, NvmmMapKeepalive):
            return keepalive
        if isinstance(keepalive, tuple) and len(keepalive) >= 2:
            surf = keepalive[1]
            existing = _KEEPALIVE.get(id(surf))
            if existing is not None:
                return existing
            ka = cls(keepalive[0], keepalive[1])
            _KEEPALIVE[id(surf)] = ka
            return ka
        return None

    def ref(self) -> "NvmmMapKeepalive":
        with self._lock:
            self._refs += 1
        return self

    def unref(self) -> None:
        surf = None
        sid = None
        with self._lock:
            self._refs -= 1
            if self._refs <= 0:
                surf = self.surf
                sid = id(surf) if surf is not None else None
                self.surf = None
                self.sample = None
        if sid is not None:
            _KEEPALIVE.pop(sid, None)
        if surf is not None:
            try:
                from .nvbuf_util import nvbuf_surface_unmap

                nvbuf_surface_unmap(surf, 0, -1)
            except Exception:
                pass


def _release_keepalive(keepalive: Any) -> None:
    if isinstance(keepalive, GstSampleKeepalive):
        keepalive.unref()
        return
    ka = NvmmMapKeepalive.adopt(keepalive)
    if ka is not None:
        ka.unref()


@dataclass
class Nv12SbsFrame:
    eyes: dict[str, "Nv12Frame"]
    width: int
    height: int
    hw_crop: bool = True
    capture_seq: int = -1
    full: "Nv12Frame | None" = None
    preview_bgr: np.ndarray | None = None
    _keepalive: Any = None
    _display_keepalive: Any = None

    def release_eyes(self) -> None:
        for eye in self.eyes.values():
            eye.release()
        self.eyes.clear()
        self._keepalive = None

    def release_full(self) -> None:
        if self.full is not None:
            self.full.release()
            self.full = None

    def release_preview(self) -> None:
        if self._display_keepalive is not None:
            _release_keepalive(self._display_keepalive)
            self._display_keepalive = None
        self.preview_bgr = None

    def release(self) -> None:
        self.release_eyes()
        self.release_full()
        self._keepalive = None
        self.release_preview()


@dataclass
class Nv12Frame:
    y: np.ndarray | None
    uv: np.ndarray | None
    width: int
    height: int
    dmabuf_fd: int | None = None
    pitch: int | None = None
    zero_copy: bool = False
    _keepalive: Any = None

    @classmethod
    def from_pitch_planes(
        cls,
        y_plane: np.ndarray,
        uv_plane: np.ndarray,
        width: int,
        height: int,
        *,
        copy: bool = True,
        dmabuf_fd: int | None = None,
        zero_copy: bool = False,
        pitch: int | None = None,
        keepalive: Any = None,
    ) -> "Nv12Frame":
        from .nvbuf_util import nvbuf_pitch_vic_compatible

        w, h = int(width), int(height)
        p = int(pitch if pitch is not None else w)
        if zero_copy and not nvbuf_pitch_vic_compatible(p):
            raise ValueError(f"NVMM pitch={p} not VIC-aligned")
        if copy:
            y = np.array(y_plane[:h, :w], dtype=np.uint8, copy=True)
            uv = np.array(
                uv_plane[: h // 2, :w].reshape(h // 2, w // 2, 2),
                dtype=np.uint8,
                copy=True,
            )
            p = w
        else:
            y = y_plane[:h, :p]
            uv = uv_plane[: h // 2, :p]
        return cls(
            y=y,
            uv=uv,
            width=w,
            height=h,
            dmabuf_fd=dmabuf_fd,
            pitch=p,
            zero_copy=zero_copy,
            _keepalive=keepalive,
        )

    def release(self) -> None:
        if self._keepalive is None:
            return
        keep = self._keepalive
        self._keepalive = None
        _release_keepalive(keep)

    def to_bgr_cpu(self) -> np.ndarray:
        """预览用 CPU BGR（仅显示路径，非采集）。"""
        from .nv12_util import nv12_er_planes_to_bgr

        h, w = self.height, self.width
        if self.y is None or self.uv is None:
            raise RuntimeError("Nv12Frame planes not mapped")
        p = int(self.pitch if self.pitch is not None else w)
        y = self.y[:h, :p]
        uv = self.uv[: h // 2, :p]
        return nv12_er_planes_to_bgr(y, uv, h, w, pitch=p)
