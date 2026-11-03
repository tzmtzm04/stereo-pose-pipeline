"""Session MJPEG index → NVDEC（专用硬件）→ NVMM NV12；后处理走 VIC，不用 GPU。"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from contracts.session_bundle import SessionBundle
from core.offline.session_decoder import load_index_rows, load_session_meta
from platform.jetson.gst_capture import gst_start_playing, pull_sample_timeout
from platform.jetson.gst_nvmm_capture import _nv12_from_sample, gst_nvmm_available
from platform.jetson.nv12_frame import Nv12Frame, gst_sample_keepalive_acquire

logger = logging.getLogger(__name__)


def session_nvdec_available() -> bool:
    return gst_nvmm_available()


@dataclass(frozen=True)
class SessionNv12Frame:
    capture_seq: int
    sbs_nv12: Nv12Frame

    def release(self) -> None:
        self.sbs_nv12.release()


class SessionNvdecDecoder:
    """JPEG → nvv4l2decoder(NVDEC) → nvvidconv(VIC) → NVMM NV12 appsink。"""

    # Jetson nvvidconv: 0=Default(VIC), 1=GPU, 2=VIC
    _NVVIDCONV_VIC = "nvvidconv compute-hw=2"

    def __init__(
        self,
        width: int,
        height: int,
        *,
        fps: int = 60,
        appsink_max_buffers: int = 1,
    ) -> None:
        if not session_nvdec_available():
            raise RuntimeError(
                "NVDEC 离线解码需要 GStreamer + NVMM；"
                "回退: --cpu 或检查 Jetson GStreamer 插件"
            )
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        self._Gst = Gst
        self.width = int(width)
        self.height = int(height)
        self.fps = max(1, int(fps))
        self._pts = 0
        self._duration = Gst.SECOND // self.fps
        sink_bufs = max(1, int(appsink_max_buffers))
        caps = (
            f"image/jpeg,width={self.width},height={self.height},"
            f"framerate={self.fps}/1"
        )
        pipeline_str = (
            f"appsrc name=src is-live=true format=time do-timestamp=true caps={caps} ! "
            f"jpegparse ! "
            f"nvv4l2decoder mjpeg=1 ! {self._NVVIDCONV_VIC} ! "
            f"video/x-raw(memory:NVMM),width={self.width},height={self.height},format=NV12 ! "
            f"{self._NVVIDCONV_VIC} bl-output=false ! "
            f"appsink name=sink emit-signals=false sync=false max-buffers={sink_bufs} drop=false"
        )
        self._pipeline = Gst.parse_launch(pipeline_str)
        self._appsrc = self._pipeline.get_by_name("src")
        self._appsink = self._pipeline.get_by_name("sink")
        if self._appsrc is None or self._appsink is None:
            raise RuntimeError(f"NVDEC pipeline 元素缺失: {pipeline_str}")
        gst_start_playing(self._pipeline, Gst, pull_driven=True)

    def decode_jpeg(self, jpeg_bytes: bytes) -> Nv12Frame | None:
        Gst = self._Gst
        if not jpeg_bytes:
            return None
        buf = Gst.Buffer.new_allocate(None, len(jpeg_bytes), None)
        buf.fill(0, jpeg_bytes)
        buf.pts = self._pts
        buf.duration = self._duration
        self._pts += self._duration
        flow = self._appsrc.emit("push-buffer", buf)
        if flow != Gst.FlowReturn.OK:
            logger.warning("appsrc push-buffer failed: %s", flow)
            return None
        sample = pull_sample_timeout(self._appsink, Gst, timeout_sec=2.0)
        if sample is None:
            logger.warning("NVDEC pull sample timeout")
            return None
        ka = gst_sample_keepalive_acquire(sample)
        return _nv12_from_sample(
            sample,
            Gst,
            map_planes=False,
            sample_keepalive=ka,
        )

    def close(self) -> None:
        Gst = self._Gst
        if self._appsrc is not None:
            try:
                self._appsrc.emit("end-of-stream")
            except Exception:
                pass
        if self._pipeline is not None:
            self._pipeline.set_state(Gst.State.NULL)
        self._appsrc = None
        self._appsink = None
        self._pipeline = None


def iter_session_nv12_frames(
    bundle: SessionBundle,
    *,
    start_capture_seq: int | None = None,
    end_capture_seq: int | None = None,
    stride: int = 1,
    width: int | None = None,
    height: int | None = None,
    fps: int | None = None,
) -> Iterator[SessionNv12Frame]:
    """按 index 顺序 NVDEC 解码 session.mjpeg → NVMM NV12 SBS fd。"""
    meta = load_session_meta(bundle.session_dir)
    w = int(width if width is not None else meta.get("width", 3840) or 3840)
    h = int(height if height is not None else meta.get("height", 1080) or 1080)
    f = int(fps if fps is not None else meta.get("fps", 60) or 60)

    rows = load_index_rows(
        bundle,
        start_capture_seq=start_capture_seq,
        end_capture_seq=end_capture_seq,
    )
    if int(stride) > 1:
        rows = rows[:: int(stride)]

    decoder = SessionNvdecDecoder(w, h, fps=f)
    mjpeg_path = bundle.mjpeg_path
    try:
        with mjpeg_path.open("rb") as fp:
            for row in rows:
                fp.seek(int(row.offset))
                data = fp.read(int(row.size))
                if not data:
                    continue
                nv12 = decoder.decode_jpeg(data)
                if nv12 is None:
                    continue
                yield SessionNv12Frame(
                    capture_seq=int(row.capture_seq),
                    sbs_nv12=nv12,
                )
    finally:
        decoder.close()
