"""
GStreamer UVC → NVMM NV12（零拷贝 dmabuf，MotionLabs 同思路）。

MJPEG → nvv4l2decoder → nvvidconv → NVMM NV12；
双目 SBS 由 nvvidconv 硬件裁左右眼（VIC L2）。
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from typing import Any, Callable

from contracts.seq_util import int_attr

from .gst_capture import (
    device_held_by_current_process,
    gst_available,
    gst_drain_bus_errors,
    gst_force_pipeline_null,
    gst_null_v4l2src,
    gst_start_playing,
    gst_stop_pipeline,
    normalize_device,
    pull_sample_timeout,
    GST_PLAY_TIMEOUT_SEC,
    GST_STOP_TIMEOUT_SEC,
)
from .nv12_frame import (
    GstSampleKeepalive,
    Nv12Frame,
    Nv12SbsFrame,
    NvmmMapKeepalive,
    gst_sample_keepalive_acquire,
)
from .nvbuf_util import (
    NVBUF_LAYOUT_PITCH,
    NVBUF_MAP_READ,
    nvbuf_available,
    nvbuf_require_vic_pitch,
    nvbuf_surface_from_fd,
    nvbuf_surface_from_gst_map,
    nvbuf_surface_map,
    nvbuf_surface_sync_for_cpu,
    nvbuf_surface_unmap,
    nv12_plane_arrays,
)

logger = logging.getLogger(__name__)


def _seq_audit_enabled() -> bool:
    return os.environ.get("ROUTE2_SEQ_AUDIT", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )


def _seq_audit_every() -> int:
    raw = os.environ.get("ROUTE2_SEQ_AUDIT_EVERY", "60").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 60


def gst_nvmm_available() -> bool:
    return gst_available() and nvbuf_available()


def _nv12_from_sample(
    sample: Any,
    Gst: Any,
    *,
    map_planes: bool = True,
    copy: bool = False,
    sample_keepalive: GstSampleKeepalive | None = None,
) -> Nv12Frame:
    buf = sample.get_buffer()
    ok, map_info = buf.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError("GstBuffer map failed")
    surf = None
    try:
        gst_surf = nvbuf_surface_from_gst_map(map_info.data)
        fd = int(gst_surf.contents.surfaceList[0].bufferDesc)
        surf = nvbuf_surface_from_fd(fd)
        layout = int(surf.contents.surfaceList[0].layout)
        if layout != NVBUF_LAYOUT_PITCH:
            raise RuntimeError(
                f"NVMM layout={layout}; pipeline needs nvvidconv bl-output=false"
            )
        pitch = nvbuf_require_vic_pitch(surf, 0)
        w = int_attr(surf.contents.surfaceList[0], "width", default=0)
        h = int_attr(surf.contents.surfaceList[0], "height", default=0)
        if not map_planes:
            ka = sample_keepalive or gst_sample_keepalive_acquire(sample)
            return Nv12Frame(
                y=None,
                uv=None,
                width=w,
                height=h,
                dmabuf_fd=fd,
                zero_copy=True,
                pitch=pitch,
                _keepalive=ka,
            )
        nvbuf_surface_map(surf, 0, -1, NVBUF_MAP_READ)
        nvbuf_surface_sync_for_cpu(surf, 0, -1)
        y_plane, uv_plane, w, h, pitch = nv12_plane_arrays(surf, 0)
        return Nv12Frame.from_pitch_planes(
            y_plane,
            uv_plane,
            w,
            h,
            copy=copy,
            dmabuf_fd=fd,
            zero_copy=not copy,
            pitch=pitch,
            keepalive=NvmmMapKeepalive(sample, surf),
        )
    except Exception:
        if surf is not None:
            nvbuf_surface_unmap(surf, 0, -1)
        raise
    finally:
        buf.unmap(map_info)


def _v4l2_mjpeg_src(device: str, width: int, height: int, fps: int) -> str:
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
    )


def _decoder_props(*, decoder_extra_surfaces: int = 0, name: str = "") -> str:
    extra = max(0, int(decoder_extra_surfaces))
    dec_props = "mjpeg=1"
    if name:
        dec_props += f" name={name}"
    if extra > 0:
        dec_props += f" num-extra-surfaces={extra}"
    return dec_props


def _decoder_chain(
    device: str,
    width: int,
    height: int,
    fps: int,
    *,
    decoder_extra_surfaces: int = 0,
) -> str:
    extra = max(0, int(decoder_extra_surfaces))
    dec_props = _decoder_props(decoder_extra_surfaces=extra)
    return (
        f"v4l2src device={device} io-mode=2 ! "
        f"image/jpeg,width={width},height={height},framerate={fps}/1 ! "
        f"nvv4l2decoder {dec_props} ! nvvidconv ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},format=NV12 ! "
        f"nvvidconv bl-output=false ! "
    )


class JetsonGstNvmmCapture:
    """NVMM NV12 采集；hw_sbs_crop 时左右眼各 1920×1080 NVMM fd。"""

    def __init__(
        self,
        camera: str | int = "/dev/video0",
        width: int = 3840,
        height: int = 1080,
        fps: int = 60,
        *,
        sbs_mode: str = "both",
        hw_sbs_crop: bool = False,
        ring_depth: int = 2,
        enc_sbs_branch: bool = False,
        pull_full_in_read: bool = False,
        capture_vic_split: bool = False,
        decoder_extra_surfaces: int = 0,
        sbs_tee: bool = True,
        display_sbs_branch: bool = False,
        display_full_sbs: bool = False,
        display_wait_pose: bool = False,
        mjpeg_split: bool = True,
        mjpeg_rec_queue_depth: int = 64,
    ) -> None:
        if not gst_nvmm_available():
            raise RuntimeError("需要 GStreamer (gi) + libnvbufsurface")
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        self._Gst = Gst
        self.sbs_mode = sbs_mode
        self.hw_sbs_crop = hw_sbs_crop
        self.ring_depth = max(1, int(ring_depth))
        self.enc_sbs_branch = bool(enc_sbs_branch)
        self.pull_full_in_read = bool(pull_full_in_read)
        self.capture_vic_split = bool(capture_vic_split)
        self._sbs_tee = bool(sbs_tee)
        self.display_sbs_branch = bool(display_sbs_branch)
        self.display_full_sbs = bool(display_full_sbs)
        self._display_wait_pose = bool(display_wait_pose)
        self._mjpeg_split = bool(mjpeg_split)
        self._mjpeg_rec_queue_depth = max(8, int(mjpeg_rec_queue_depth))
        self._mjpeg_ingress_handler: Callable[[int, Any], None] | None = None
        self._sbs_splitter: Any | None = None
        self._decoder_pending: dict[str, deque[int]] = {}
        self._decoder_pending_lock = threading.Lock()
        device = normalize_device(camera)
        self._device = device
        self.width = int(width)
        self.height = int(height)
        self.eye_w = self.width // 2
        self.eye_h = self.height
        self.decoder_extra_surfaces = max(0, int(decoder_extra_surfaces))

        dec = _decoder_chain(
            device,
            self.width,
            self.height,
            fps,
            decoder_extra_surfaces=self.decoder_extra_surfaces,
        )
        d = self.ring_depth
        if self._mjpeg_split:
            pipeline_str = self._pipeline_mjpeg_split_b(device, fps=fps, ring_depth=d)
        elif capture_vic_split and sbs_mode in ("left", "right", "both"):
            pipeline_str = self._pipeline_vic_split_capture(dec, ring_depth=d)
        elif hw_sbs_crop and sbs_mode in ("left", "right", "both"):
            pipeline_str = self._pipeline_hw_crop(dec, sbs_mode, ring_depth=d)
        elif sbs_mode in ("left", "right", "both"):
            if self._sbs_tee:
                pipeline_str = self._pipeline_sbs_single(dec, ring_depth=d)
            else:
                pipeline_str = self._pipeline_sbs_direct(dec, ring_depth=d)
        else:
            pipeline_str = (
                dec
                + f"video/x-raw(memory:NVMM),width={self.width},height={self.height},format=NV12 ! "
                f"appsink name=sink emit-signals=false max-buffers={d} drop=true sync=false "
                "async=false enable-last-sample=false"
            )

        logger.info("GStreamer NVMM pipeline: %s", pipeline_str)
        print(f"Capture: GStreamer NVMM NV12 zero-copy\n  {pipeline_str}")
        self.pipeline = Gst.parse_launch(pipeline_str)
        self.sink = self.pipeline.get_by_name("sink")
        self.sink_left = self.pipeline.get_by_name("sink_left")
        self.sink_right = self.pipeline.get_by_name("sink_right")
        self.sink_sbs_full = self.pipeline.get_by_name("sink_sbs_full")
        self.sink_capture_sbs = self.pipeline.get_by_name("sink_capture_sbs")
        self.sink_display_left = self.pipeline.get_by_name("sink_display_left")
        self.sink_mjpeg_record = self.pipeline.get_by_name("sink_mjpeg_record")
        self._hw_crop = (
            hw_sbs_crop
            and sbs_mode in ("left", "right", "both")
            and not capture_vic_split
            and not self._mjpeg_split
        )
        self._vic_split_capture = (
            capture_vic_split and sbs_mode == "both" and not self._mjpeg_split
        )
        self._sbs_single = (
            sbs_mode in ("left", "right", "both")
            and not self._hw_crop
            and not self._vic_split_capture
        )
        if self._mjpeg_split:
            self._sbs_tee = False
        self._playing = False
        self._read_fail_streak = 0
        self._pull_timeout_sec = 0.5
        # display 生产线：空队列时勿阻塞 500ms，否则 UI 假死（~2fps）
        self._display_pull_timeout_sec = 0.05
        self._capture_seq = 0
        self._display_seq_fallback = 0
        self._display_drained = False
        self._last_capture_pull_seq = -1
        self._latest_ingress_seq = -1
        self._ingress_seq = 0
        self._ingress_probe_id: int | None = None
        self._seq_audit = _seq_audit_enabled()
        self._seq_audit_every = _seq_audit_every()
        self._seq_audit_counts: dict[str, int] = {}
        self._seq_audit_last_pull: dict[str, int] = {}
        self._seq_audit_bridge_miss = 0
        if self._seq_audit:
            print(
                f"seq_audit 开启 every={self._seq_audit_every} "
                f"(ROUTE2_SEQ_AUDIT=1；关则 unset)",
                flush=True,
            )
        if self._mjpeg_split:
            self._attach_mjpeg_tee_ingress_probe()
            if self.display_sbs_branch:
                self._attach_decoder_tee_seq_bridge("infer_dec", "sbs_tee")
            else:
                self._attach_decoder_seq_bridge("infer_dec", "sink_capture_sbs")
        elif self._hw_crop and self.sbs_mode == "both":
            self._attach_sbs_tee_ingress_probe()
        elif self._vic_split_capture or self._sbs_single:
            if self._sbs_tee:
                self._attach_sbs_tee_ingress_probe()
            else:
                self._attach_capture_sbs_ingress_probe()

    def set_sbs_vic_splitter(self, splitter: Any) -> None:
        self._sbs_splitter = splitter

    def _seq_audit_note(self, tag: str, seq: int, **fields: Any) -> None:
        if not self._seq_audit:
            return
        n = int(self._seq_audit_counts.get(tag, 0)) + 1
        self._seq_audit_counts[tag] = n
        if n != 1 and n % self._seq_audit_every != 0:
            return
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        msg = f"seq_audit {tag} seq={int(seq)} n={n}"
        if extra:
            msg = f"{msg} {extra}"
        print(msg, flush=True)

    def _seq_audit_warn(self, tag: str, **fields: Any) -> None:
        if not self._seq_audit:
            return
        self._seq_audit_bridge_miss += 1
        n = self._seq_audit_bridge_miss
        if n != 1 and n % self._seq_audit_every != 0:
            return
        extra = " ".join(f"{k}={v}" for k, v in fields.items())
        msg = f"seq_audit WARN {tag}"
        if extra:
            msg = f"{msg} {extra}"
        print(msg, flush=True)

    def _record_pull(self, branch: str, seq: int, *, ingress_seq: int | None) -> None:
        seq = int(seq)
        if branch == "capture":
            self._last_capture_pull_seq = seq
        self._seq_audit_last_pull[str(branch)] = seq
        if not self._seq_audit:
            return
        other = "capture" if branch == "display" else "display"
        other_seq = self._seq_audit_last_pull.get(other)
        delta = (
            int(seq) - int(other_seq)
            if other_seq is not None
            else None
        )
        self._seq_audit_note(
            f"pull_{branch}",
            seq,
            buf_offset=(int(ingress_seq) + 1) if ingress_seq is not None else -1,
            pair_delta=delta if delta is not None else "na",
        )

    def _seq_audit_pull(self, branch: str, seq: int, *, ingress_seq: int | None) -> None:
        self._record_pull(branch, seq, ingress_seq=ingress_seq)

    def _appsink_props(self, ring_depth: int) -> str:
        d = max(1, int(ring_depth))
        return (
            f"emit-signals=false max-buffers={d} drop=true sync=false "
            "async=false enable-last-sample=false"
        )

    def _display_tee_queue_depth(self, ring_depth: int) -> int:
        """与 capture 同深，避免 display 侧额外 backlog 造成固定帧差。"""
        return max(1, int(ring_depth))

    def _display_tee_queue_line(self, ring_depth: int) -> str:
        q = self._display_tee_queue_depth(ring_depth)
        return f"max-size-buffers={q} leaky=downstream ! "

    def _display_appsink_props(self, ring_depth: int) -> str:
        """display appsink 一律 drop=true，避免阻塞 mjpeg_tee（录像优先）。"""
        q = self._display_tee_queue_depth(ring_depth)
        return (
            f"emit-signals=false max-buffers={q} drop=true sync=false "
            "async=false enable-last-sample=false"
        )

    def _pipeline_vic_split_capture(self, dec: str, *, ring_depth: int) -> str:
        """全幅 SBS pull + 裸帧环；左右眼 VIC 对半裁在 Python 采集侧。"""
        full = self.width
        eh = self.height
        d = max(1, int(ring_depth))
        sink = self._appsink_props(d)
        ring_branch = ""
        if self.enc_sbs_branch:
            ring_branch = (
                f"sbs_tee. ! queue max-size-buffers={d} leaky=downstream ! "
                f"video/x-raw(memory:NVMM),width={full},height={eh},format=NV12 ! "
                f"appsink name=sink_sbs_full {sink} "
            )
        return (
            dec
            + "tee name=sbs_tee "
            + ring_branch
            + f"sbs_tee. ! queue max-size-buffers={d} leaky=downstream ! "
            f"video/x-raw(memory:NVMM),width={full},height={eh},format=NV12 ! "
            f"appsink name=sink_capture_sbs {sink}"
        )

    @property
    def capture_seq(self) -> int:
        """下一帧将分配的 capture_seq（已成功 read 的次数）。"""
        return self._capture_seq

    def _mjpeg_rec_sink_props(self) -> str:
        q = self._mjpeg_rec_queue_depth
        return (
            f"emit-signals=false max-buffers={q} drop=false sync=false "
            "async=false enable-last-sample=false"
        )

    def _nvdec_nv12_appsink_chain(
        self,
        *,
        dec_name: str,
        width: int,
        height: int,
        post_nvvidconv: str = "",
        sink_name: str,
        sink_props: str,
    ) -> str:
        dec_props = _decoder_props(
            decoder_extra_surfaces=self.decoder_extra_surfaces,
            name=dec_name,
        )
        chain = (
            f"nvv4l2decoder {dec_props} ! nvvidconv ! "
            f"video/x-raw(memory:NVMM),width={width},height={height},format=NV12 ! "
            f"nvvidconv bl-output=false"
        )
        if post_nvvidconv:
            post = post_nvvidconv.strip().rstrip("!").strip()
            chain += f" {post} ! "
        else:
            chain += " ! "
        return chain + f"appsink name={sink_name} {sink_props}"

    def _pipeline_mjpeg_split_b(
        self,
        device: str,
        *,
        fps: int,
        ring_depth: int,
    ) -> str:
        """形态 B：mjpeg_tee → ingress 录像 + 单次 NVDEC → NV12 sbs_tee → infer / display。"""
        src = _v4l2_mjpeg_src(device, self.width, self.height, fps)
        d = max(1, int(ring_depth))
        sink = self._appsink_props(d)
        full = self.width
        eh = self.height
        rec_q = self._mjpeg_rec_queue_depth
        dec_props = _decoder_props(
            decoder_extra_surfaces=self.decoder_extra_surfaces,
            name="infer_dec",
        )
        decode_prefix = (
            f"mjpeg_tee. ! queue name=mjpeg_infer_q max-size-buffers={d} "
            f"leaky=downstream ! "
            f"nvv4l2decoder {dec_props} ! nvvidconv ! "
            f"video/x-raw(memory:NVMM),width={full},height={eh},format=NV12 ! "
            f"nvvidconv bl-output=false ! "
        )
        parts = [src + "tee name=mjpeg_tee "]
        if self.display_sbs_branch:
            # 形态 B：display 必须全幅 SBS（tee 后无 nvvidconv），否则 offset 丢失
            display_branch = self._display_full_sbs_branch(ring_depth=d)
            parts.append(
                decode_prefix
                + "tee name=sbs_tee "
                + display_branch
                + f"sbs_tee. ! queue max-size-buffers={d} leaky=downstream ! "
                f"video/x-raw(memory:NVMM),width={full},height={eh},format=NV12 ! "
                f"appsink name=sink_capture_sbs {sink}"
            )
        else:
            parts.append(
                decode_prefix
                + f"video/x-raw(memory:NVMM),width={full},height={eh},format=NV12 ! "
                f"appsink name=sink_capture_sbs {sink}"
            )
        return " ".join(part.strip() for part in parts if part.strip())

    @property
    def uses_mjpeg_split(self) -> bool:
        return self._mjpeg_split

    def set_mjpeg_ingress_recorder(
        self,
        handler: Callable[[int, Any], None] | None,
    ) -> None:
        """ZED 对齐：mjpeg_tee ingress 回调 (capture_seq, GstBuffer)，probe 内须非阻塞。"""
        self._mjpeg_ingress_handler = handler

    def mjpeg_buffer_payload(self, buf: Any) -> bytes | None:
        """从 GstBuffer 读取 MJPEG 字节（ingress worker 后台线程调用）。"""
        return self._mjpeg_buffer_payload(buf)

    def _mjpeg_buffer_payload(self, buf: Any) -> bytes | None:
        """v4l2 MJPEG：extract_dup → 逐 memory map → 整 buf map（含 COPY）。"""
        Gst = self._Gst
        size = int(buf.get_size())
        if size <= 0:
            return None
        try:
            dup = buf.extract_dup(0, size)
            if dup:
                return bytes(dup)
        except Exception:
            pass
        chunks: list[bytes] = []
        n_mem = int(buf.n_memory())
        if n_mem > 0:
            for idx in range(n_mem):
                mem = buf.peek_memory(idx)
                ok, map_info = mem.map(Gst.MapFlags.READ)
                if not ok:
                    continue
                try:
                    n = int(map_info.size)
                    if n > 0:
                        chunks.append(bytes(map_info.data[:n]))
                finally:
                    mem.unmap(map_info)
            if chunks:
                return b"".join(chunks) if len(chunks) > 1 else chunks[0]
        map_flags = Gst.MapFlags.READ
        copy_flag = getattr(Gst.MapFlags, "COPY", None)
        if copy_flag is not None:
            map_flags = map_flags | copy_flag
        ok, map_info = buf.map(map_flags)
        if not ok:
            return None
        try:
            return bytes(map_info.data[: map_info.size])
        finally:
            buf.unmap(map_info)

    def _attach_mjpeg_tee_ingress_probe(self) -> None:
        tee = self.pipeline.get_by_name("mjpeg_tee")
        if tee is None:
            return
        sink_pad = tee.get_static_pad("sink")
        if sink_pad is None:
            return
        Gst = self._Gst

        def _probe(_pad: Any, info: Any) -> Any:
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK
            seq = self._ingress_seq
            self._ingress_seq += 1
            self._latest_ingress_seq = int(seq)
            buf.offset = seq + 1
            self._seq_audit_note("mjpeg_ingress", int(seq), offset=int(seq) + 1)
            handler = self._mjpeg_ingress_handler
            if handler is not None:
                try:
                    handler(seq, buf)
                except Exception as exc:
                    logger.warning("mjpeg ingress recorder: %s", exc)
            return Gst.PadProbeReturn.OK

        self._ingress_probe_id = sink_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            _probe,
        )

    def _attach_decoder_seq_bridge(self, decoder_name: str, appsink_name: str) -> None:
        """JPEG buffer.offset → NV12 appsink buffer.offset（decoder 输出不保留 offset）。"""
        dec = self.pipeline.get_by_name(decoder_name)
        appsink = self.pipeline.get_by_name(appsink_name)
        if dec is None or appsink is None:
            return
        dec_sink = dec.get_static_pad("sink")
        app_sink = appsink.get_static_pad("sink")
        if dec_sink is None or app_sink is None:
            return
        pending: deque[int] = deque()
        self._decoder_pending[decoder_name] = pending
        Gst = self._Gst

        def _dec_probe(_pad: Any, info: Any) -> Any:
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK
            seq = self._capture_seq_from_buffer(buf)
            if seq is not None:
                with self._decoder_pending_lock:
                    pending.append(int(seq))
                    depth = len(pending)
                self._seq_audit_note("dec_in", int(seq), pending=depth)
            else:
                self._seq_audit_warn("dec_in_missing_offset")
            return Gst.PadProbeReturn.OK

        def _app_probe(_pad: Any, info: Any) -> Any:
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK
            with self._decoder_pending_lock:
                if pending:
                    seq = pending.popleft()
                    buf.offset = seq + 1
                    depth = len(pending)
                    self._seq_audit_note("nv12_appsink", int(seq), pending=depth)
                else:
                    self._seq_audit_warn("nv12_appsink_empty_pending")
            return Gst.PadProbeReturn.OK

        dec_sink.add_probe(Gst.PadProbeType.BUFFER, _dec_probe)
        app_sink.add_probe(Gst.PadProbeType.BUFFER, _app_probe)

    def _attach_decoder_tee_seq_bridge(self, decoder_name: str, tee_name: str) -> None:
        """JPEG buffer.offset → NV12 sbs_tee 入口 buffer.offset（infer/display 共享序号）。"""
        dec = self.pipeline.get_by_name(decoder_name)
        tee = self.pipeline.get_by_name(tee_name)
        if dec is None or tee is None:
            return
        dec_sink = dec.get_static_pad("sink")
        tee_sink = tee.get_static_pad("sink")
        if dec_sink is None or tee_sink is None:
            return
        pending: deque[int] = deque()
        self._decoder_pending[decoder_name] = pending
        Gst = self._Gst

        def _dec_probe(_pad: Any, info: Any) -> Any:
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK
            seq = self._capture_seq_from_buffer(buf)
            if seq is not None:
                with self._decoder_pending_lock:
                    pending.append(int(seq))
                    depth = len(pending)
                self._seq_audit_note("dec_in", int(seq), pending=depth)
            else:
                self._seq_audit_warn("dec_in_missing_offset")
            return Gst.PadProbeReturn.OK

        def _tee_probe(_pad: Any, info: Any) -> Any:
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK
            with self._decoder_pending_lock:
                if pending:
                    seq = pending.popleft()
                    buf.offset = seq + 1
                    depth = len(pending)
                    self._seq_audit_note("sbs_tee", int(seq), pending=depth)
                else:
                    self._seq_audit_warn("sbs_tee_empty_pending")
            return Gst.PadProbeReturn.OK

        dec_sink.add_probe(Gst.PadProbeType.BUFFER, _dec_probe)
        tee_sink.add_probe(Gst.PadProbeType.BUFFER, _tee_probe)

    def pull_mjpeg_record(
        self,
        *,
        timeout_sec: float | None = None,
    ) -> tuple[bool, bytes | None, int]:
        """pull MJPEG 录像支路；返回 (ok, jpeg_bytes, capture_seq)。"""
        self._ensure_playing()
        Gst = self._Gst
        sink = self.sink_mjpeg_record
        if sink is None:
            return False, None, -1
        t = self._pull_timeout_sec if timeout_sec is None else float(timeout_sec)
        sample = pull_sample_timeout(sink, Gst, timeout_sec=t)
        if sample is None:
            return False, None, -1
        try:
            buf = sample.get_buffer()
            if buf is None:
                return False, None, -1
            ok, map_info = buf.map(Gst.MapFlags.READ)
            if not ok:
                return False, None, -1
            try:
                data = bytes(map_info.data[: map_info.size])
            finally:
                buf.unmap(map_info)
            ingress_seq = self._capture_seq_from_buffer(buf)
            seq = int(ingress_seq if ingress_seq is not None else -1)
            return True, data, seq
        except Exception:
            return False, None, -1

    def _attach_sbs_tee_ingress_probe(self) -> None:
        """sbs_tee sink：每 ingress 帧 +1，写入 buffer.offset（seq+1，0 表示未标）。"""
        tee = self.pipeline.get_by_name("sbs_tee")
        if tee is None:
            return
        sink_pad = tee.get_static_pad("sink")
        if sink_pad is None:
            return
        Gst = self._Gst

        def _probe(_pad: Any, info: Any) -> Any:
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK
            seq = self._ingress_seq
            self._ingress_seq += 1
            buf.offset = seq + 1
            return Gst.PadProbeReturn.OK

        self._ingress_probe_id = sink_pad.add_probe(
            Gst.PadProbeType.BUFFER,
            _probe,
        )

    def _capture_seq_from_buffer(self, buf: Any) -> int | None:
        offset = int(buf.offset)
        if offset > 0:
            return offset - 1
        return None

    def _stamp_capture_seq(
        self,
        frame: Nv12Frame | Nv12SbsFrame | None,
        *,
        ingress_seq: int | None = None,
    ) -> None:
        if frame is None:
            return
        if ingress_seq is not None:
            seq = int(ingress_seq)
        else:
            seq = int(self._capture_seq)
            self._capture_seq += 1
        if isinstance(frame, Nv12SbsFrame):
            frame.capture_seq = seq

    @property
    def record_tee_name(self) -> str | None:
        """全幅 SBS tee 元素名（both + sbs_single / hw_crop / vic_split）。"""
        if self._mjpeg_split:
            return None
        if not self._sbs_tee:
            return None
        if self.sbs_mode == "both" and (
            self._hw_crop or self._vic_split_capture or self._sbs_single
        ):
            return "sbs_tee"
        return None

    def _pipeline_sbs_direct(self, dec: str, *, ring_depth: int) -> str:
        """全幅 SBS 单路 appsink（GrabRuntime，无 tee）。"""
        full = self.width
        eh = self.height
        d = max(1, int(ring_depth))
        sink = self._appsink_props(d)
        return (
            dec
            + f"video/x-raw(memory:NVMM),width={full},height={eh},format=NV12 ! "
            f"appsink name=sink_capture_sbs {sink}"
        )

    def _attach_capture_sbs_ingress_probe(self) -> None:
        """单路 appsink sink：每 ingress 帧 +1，写入 buffer.offset。"""
        sink = self.sink_capture_sbs
        if sink is None:
            return
        pad = sink.get_static_pad("sink")
        if pad is None:
            return
        Gst = self._Gst

        def _probe(_pad: Any, info: Any) -> Any:
            buf = info.get_buffer()
            if buf is None:
                return Gst.PadProbeReturn.OK
            seq = self._ingress_seq
            self._ingress_seq += 1
            buf.offset = seq + 1
            return Gst.PadProbeReturn.OK

        self._ingress_probe_id = pad.add_probe(
            Gst.PadProbeType.BUFFER,
            _probe,
        )

    def _display_left_branch(self, *, ring_depth: int) -> str:
        """display 独立支路：VIC 裁左眼（imshow / CPU 预览）。"""
        ew, eh = self.eye_w, self.eye_h
        d = max(1, int(ring_depth))
        sink = self._display_appsink_props(d)
        return (
            f"sbs_tee. ! queue {self._display_tee_queue_line(d)}"
            f"nvvidconv bl-output=false left=0 top=0 right={ew} bottom={eh} ! "
            f"video/x-raw(memory:NVMM),width={ew},height={eh},format=NV12 ! "
            f"appsink name=sink_display_left {sink} "
        )

    def _display_full_sbs_branch(self, *, ring_depth: int) -> str:
        """display 支路：全幅 SBS（无 VIC），GL/GPU 裁左眼。"""
        full = self.width
        eh = self.height
        d = max(1, int(ring_depth))
        sink = self._display_appsink_props(d)
        return (
            f"sbs_tee. ! queue {self._display_tee_queue_line(d)}"
            f"video/x-raw(memory:NVMM),width={full},height={eh},format=NV12 ! "
            f"appsink name=sink_display_left {sink} "
        )

    def _pipeline_sbs_single(self, dec: str, *, ring_depth: int) -> str:
        """全幅 SBS 单路 pull；推理 GPU 裁半，无 VIC hw_crop。"""
        full = self.width
        eh = self.height
        d = max(1, int(ring_depth))
        sink = self._appsink_props(d)
        ring_branch = ""
        if self.enc_sbs_branch:
            ring_branch = (
                f"sbs_tee. ! queue max-size-buffers={d} leaky=downstream ! "
                f"video/x-raw(memory:NVMM),width={full},height={eh},format=NV12 ! "
                f"appsink name=sink_sbs_full {sink} "
            )
        display_branch = ""
        if self.display_sbs_branch:
            if self.display_full_sbs:
                display_branch = self._display_full_sbs_branch(ring_depth=d)
            else:
                display_branch = self._display_left_branch(ring_depth=d)
        return (
            dec
            + "tee name=sbs_tee "
            + ring_branch
            + display_branch
            + f"sbs_tee. ! queue max-size-buffers={d} leaky=downstream ! "
            f"video/x-raw(memory:NVMM),width={full},height={eh},format=NV12 ! "
            f"appsink name=sink_capture_sbs {sink}"
        )

    def ensure_playing(self) -> None:
        """显式启动管线（录制需在首帧 pull 前 PLAYING）。"""
        self._ensure_playing()

    def _pipeline_hw_crop(self, dec: str, sbs_mode: str, *, ring_depth: int) -> str:
        ew, eh = self.eye_w, self.eye_h
        full = self.width
        d = max(1, int(ring_depth))
        sink = (
            f"emit-signals=false max-buffers={d} drop=true sync=false "
            "async=false enable-last-sample=false"
        )
        if sbs_mode == "left":
            return (
                dec
                + f"nvvidconv bl-output=false left=0 top=0 right={ew} bottom={eh} ! "
                f"video/x-raw(memory:NVMM),width={ew},height={eh},format=NV12 ! "
                f"appsink name=sink_left {sink}"
            )
        if sbs_mode == "right":
            return (
                dec
                + f"nvvidconv bl-output=false left={ew} top=0 right={full} bottom={eh} ! "
                f"video/x-raw(memory:NVMM),width={ew},height={eh},format=NV12 ! "
                f"appsink name=sink_right {sink}"
            )
        full_branch = ""
        if self.enc_sbs_branch or self.pull_full_in_read:
            full_branch = (
                f"sbs_tee. ! queue max-size-buffers={d} leaky=downstream ! "
                f"video/x-raw(memory:NVMM),width={full},height={eh},format=NV12 ! "
                f"appsink name=sink_sbs_full {sink} "
            )
        return (
            dec
            + "tee name=sbs_tee "
            + full_branch
            + f"sbs_tee. ! tee name=t "
            + f"t. ! queue max-size-buffers={d} leaky=downstream ! "
            f"nvvidconv bl-output=false left=0 top=0 right={ew} bottom={eh} ! "
            f"video/x-raw(memory:NVMM),width={ew},height={eh},format=NV12 ! "
            f"appsink name=sink_left {sink} "
            + f"t. ! queue max-size-buffers={d} leaky=downstream ! "
            f"nvvidconv bl-output=false left={ew} top=0 right={full} bottom={eh} ! "
            f"video/x-raw(memory:NVMM),width={ew},height={eh},format=NV12 ! "
            f"appsink name=sink_right {sink}"
        )

    def _ensure_playing(self) -> None:
        if self._playing:
            return
        pull_driven = (
            self._mjpeg_split
            or (
                self._hw_crop
                and self.sink_left is not None
                and self.sink_right is not None
            )
            or (
                (self._vic_split_capture or self._sbs_single)
                and self.sink_capture_sbs is not None
            )
            or self.sink_display_left is not None
            or self.sink_mjpeg_record is not None
        )
        try:
            gst_start_playing(
                self.pipeline,
                self._Gst,
                pull_driven=pull_driven,
            )
            self._playing = True
        except Exception:
            self.release()
            raise

    def _read_sbs_single(self) -> tuple[bool, Nv12SbsFrame | None]:
        """全幅 SBS 单路 pull（GPU 裁半推理，无 VIC split）。"""
        Gst = self._Gst
        sink = self.sink_capture_sbs
        if sink is None:
            return False, None
        sample = pull_sample_timeout(sink, Gst, timeout_sec=self._pull_timeout_sec)
        if sample is None:
            return False, None
        try:
            ingress_seq = self._capture_seq_from_buffer(sample.get_buffer())
            ka = gst_sample_keepalive_acquire(sample)
            full_frame = _nv12_from_sample(
                sample, Gst, map_planes=False, copy=False, sample_keepalive=ka
            )
            frame = Nv12SbsFrame(
                eyes={},
                width=self.width,
                height=self.height,
                hw_crop=False,
                full=full_frame,
                _keepalive=ka,
            )
            self._stamp_capture_seq(frame, ingress_seq=ingress_seq)
            if ingress_seq is not None:
                self._seq_audit_pull("capture", int(frame.capture_seq), ingress_seq=int(ingress_seq))
            else:
                self._seq_audit_warn(
                    "pull_capture_missing_offset",
                    stamped=int(frame.capture_seq),
                )
            return True, frame
        except Exception:
            return False, None

    def _read_capture_sbs_full(self) -> tuple[bool, Nv12SbsFrame | None]:
        Gst = self._Gst
        sink = self.sink_capture_sbs
        if sink is None:
            return False, None
        sample = pull_sample_timeout(sink, Gst, timeout_sec=self._pull_timeout_sec)
        if sample is None:
            return False, None
        try:
            ingress_seq = self._capture_seq_from_buffer(sample.get_buffer())
            ka = gst_sample_keepalive_acquire(sample)
            full_frame = _nv12_from_sample(
                sample, Gst, map_planes=False, copy=False, sample_keepalive=ka
            )
            frame = Nv12SbsFrame(
                eyes={},
                width=self.width,
                height=self.height,
                hw_crop=False,
                full=full_frame,
                _keepalive=ka,
            )
            self._stamp_capture_seq(frame, ingress_seq=ingress_seq)
            if ingress_seq is not None:
                self._seq_audit_pull("capture", int(frame.capture_seq), ingress_seq=int(ingress_seq))
            else:
                self._seq_audit_warn(
                    "pull_capture_missing_offset",
                    stamped=int(frame.capture_seq),
                )
            if self._sbs_splitter is not None:
                frame = self._sbs_splitter.split_sbs(frame)
            return True, frame
        except Exception:
            return False, None

    def _read_hw_crop(self) -> tuple[bool, Nv12SbsFrame | None]:
        Gst = self._Gst
        eyes: dict[str, Nv12Frame] = {}
        samples: list[Any] = []
        ingress_seq: int | None = None

        def _cleanup() -> None:
            for eye in eyes.values():
                try:
                    eye.release()
                except Exception:
                    pass
            eyes.clear()
            samples.clear()

        try:
            if self.sbs_mode in ("left", "both") and self.sink_left is not None:
                s = pull_sample_timeout(
                    self.sink_left, Gst, timeout_sec=self._pull_timeout_sec
                )
                if s is None:
                    _cleanup()
                    return False, None
                samples.append(s)
                ingress_seq = self._capture_seq_from_buffer(s.get_buffer())
                eyes["left"] = _nv12_from_sample(s, Gst, map_planes=True, copy=False)
            if self.sbs_mode in ("right", "both") and self.sink_right is not None:
                s = pull_sample_timeout(
                    self.sink_right, Gst, timeout_sec=self._pull_timeout_sec
                )
                if s is None:
                    _cleanup()
                    return False, None
                samples.append(s)
                if ingress_seq is None:
                    ingress_seq = self._capture_seq_from_buffer(s.get_buffer())
                eyes["right"] = _nv12_from_sample(s, Gst, map_planes=True, copy=False)
            if not eyes:
                _cleanup()
                return False, None
            full_frame: Nv12Frame | None = None
            if self.pull_full_in_read and self.sink_sbs_full is not None:
                s_full = pull_sample_timeout(
                    self.sink_sbs_full, Gst, timeout_sec=self._pull_timeout_sec
                )
                if s_full is None:
                    _cleanup()
                    return False, None
                samples.append(s_full)
                if ingress_seq is None:
                    ingress_seq = self._capture_seq_from_buffer(s_full.get_buffer())
                full_frame = _nv12_from_sample(
                    s_full, Gst, map_planes=False, copy=False
                )
            frame = Nv12SbsFrame(
                eyes=eyes,
                width=self.width,
                height=self.height,
                hw_crop=True,
                full=full_frame,
                _keepalive=tuple(samples),
            )
            self._stamp_capture_seq(frame, ingress_seq=ingress_seq)
            return True, frame
        except Exception:
            _cleanup()
            return False, None

    _DISPLAY_SYNC_SLACK = 0
    _DISPLAY_CATCHUP_MAX = 24
    _DISPLAY_SKIP_MAX = 8

    def _display_sync_target(self) -> int:
        """display 不应再返回比 capture 更旧的 seq（固定帧差来自 backlog）。"""
        if self._last_capture_pull_seq >= 0:
            return max(0, int(self._last_capture_pull_seq) - self._DISPLAY_SYNC_SLACK)
        if self._latest_ingress_seq >= 0:
            return max(0, int(self._latest_ingress_seq) - 2)
        return -1

    def _discard_display_sample(self, sample: Any) -> None:
        try:
            ingress_seq = self._capture_seq_from_buffer(sample.get_buffer())
            if ingress_seq is not None:
                self._display_seq_fallback = int(ingress_seq) + 1
            gst_sample_keepalive_acquire(sample).unref()
        except Exception:
            pass

    def read_display_nv12(self) -> tuple[bool, Nv12Frame | None, int]:
        """display 支路 pull NV12；跳过落后于 capture 的 backlog 帧。"""
        self._ensure_playing()
        Gst = self._Gst
        sink = self.sink_display_left
        if sink is None:
            return False, None, -1
        first_pull = not self._display_drained
        if first_pull:
            self._display_drained = True
        target = self._display_sync_target()
        max_skip = self._DISPLAY_CATCHUP_MAX if first_pull else self._DISPLAY_SKIP_MAX
        skipped = 0
        while True:
            sample = pull_sample_timeout(
                sink, Gst, timeout_sec=self._display_pull_timeout_sec
            )
            if sample is None:
                return False, None, -1
            ingress_seq = self._capture_seq_from_buffer(sample.get_buffer())
            if ingress_seq is not None:
                seq = int(ingress_seq)
                self._display_seq_fallback = seq + 1
            else:
                seq = int(self._display_seq_fallback)
                self._display_seq_fallback += 1
            if target >= 0 and seq < target and skipped < max_skip:
                self._discard_display_sample(sample)
                skipped += 1
                continue
            break
        if skipped > 0:
            print(
                f"display catch-up skip={skipped} target={target} got={seq}"
                + (" (启动)" if first_pull else ""),
                flush=True,
            )
        try:
            ka = gst_sample_keepalive_acquire(sample)
            frame = _nv12_from_sample(
                sample, Gst, map_planes=False, copy=False, sample_keepalive=ka
            )
            frame.capture_seq = seq
            if ingress_seq is not None:
                self._record_pull("display", seq, ingress_seq=int(ingress_seq))
            else:
                self._seq_audit_warn(
                    "pull_display_missing_offset",
                    fallback_seq=seq,
                    full_sbs=True,
                )
                self._record_pull("display", seq, ingress_seq=None)
            return True, frame, seq
        except Exception:
            return False, None, -1

    def read_nv12(self) -> tuple[bool, Nv12Frame | Nv12SbsFrame | None]:
        self._ensure_playing()
        Gst = self._Gst
        if self._vic_split_capture:
            ok, frame = self._read_capture_sbs_full()
        elif self._sbs_single:
            ok, frame = self._read_sbs_single()
        elif self._hw_crop:
            ok, frame = self._read_hw_crop()
        elif self.sink is not None:
            sample = pull_sample_timeout(
                self.sink, Gst, timeout_sec=self._pull_timeout_sec
            )
            if sample is None:
                ok, frame = False, None
            else:
                try:
                    frame = _nv12_from_sample(sample, Gst, map_planes=True, copy=False)
                    ok = True
                except Exception:
                    ok, frame = False, None
        else:
            ok, frame = False, None

        if ok and frame is not None:
            if not isinstance(frame, Nv12SbsFrame):
                self._stamp_capture_seq(frame)
            self._read_fail_streak = 0
            return True, frame
        self._read_fail_streak += 1
        return False, None

    def prepare_shutdown(self) -> None:
        """stop 后、release 前：缩短 pull 超时，便于 worker 快退。"""
        self._playing = False
        self._pull_timeout_sec = 0.05
        self._display_pull_timeout_sec = 0.02

    def _detach_sinks(self) -> None:
        """release 前断开 appsink 引用，避免 teardown 期间再 pull。"""
        self._playing = False
        self._pull_timeout_sec = 0.05
        self._display_pull_timeout_sec = 0.02
        self.sink = None
        self.sink_left = None
        self.sink_right = None
        self.sink_sbs_full = None
        self.sink_capture_sbs = None
        self.sink_display_left = None
        self.sink_mjpeg_record = None

    def release(self) -> bool:
        """停止 pipeline 并确认本进程不再持有 V4L2 fd。成功返回 True。"""
        if getattr(self, "pipeline", None) is None:
            return not device_held_by_current_process(self._device)
        self._detach_sinks()
        pipe = self.pipeline
        self.pipeline = None
        gst_ok = False
        for attempt in range(1, 4):
            try:
                gst_ok = gst_stop_pipeline(
                    pipe,
                    self._Gst,
                    timeout_sec=GST_STOP_TIMEOUT_SEC,
                )
                if gst_ok:
                    print("route2 GStreamer pipeline → NULL", flush=True)
            except Exception as exc:
                print(f"route2 cap.release 失败 (attempt {attempt}): {exc}", flush=True)
                gst_ok = False

            deadline = time.monotonic() + 2.5
            while time.monotonic() < deadline:
                if not device_held_by_current_process(self._device):
                    return True
                time.sleep(0.05)

            if not device_held_by_current_process(self._device):
                return True
            if attempt < 3:
                print(
                    f"WARN: release 重试 {attempt}/3（仍持有 {self._device} fd）",
                    flush=True,
                )
                try:
                    gst_null_v4l2src(pipe, self._Gst)
                    gst_force_pipeline_null(
                        pipe,
                        self._Gst,
                        wait_sec=3.0,
                    )
                except Exception:
                    pass

        if device_held_by_current_process(self._device):
            print(
                f"WARN: release 后仍持有 {self._device} fd "
                f"(gst_null={int(gst_ok)})",
                flush=True,
            )
            return False
        return gst_ok
