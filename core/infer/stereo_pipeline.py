"""NVMM 双目 SBS → GPU/VIC 预处理 → TRT → 2D pose。"""

from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from app.paths import DEFAULT_YOLO26_POSE_ENGINE, resolve_yolo26_stereo_engine
from contracts.seq_util import int_attr
from core.infer.postprocess import postprocess_v26
from platform.jetson.capture import JetsonGstNvmmCapture, gst_nvmm_available
from platform.jetson.cuda.nv12_rgb import (
    cuda_nv12_rgb_available,
    gpu_crop_letterbox_cuda_available,
    gpu_letterbox_cuda_available,
    gpu_stereo_sbs_letterbox_cuda_available,
)
from platform.jetson.letterbox_util import LetterboxGeometry
from platform.jetson.nv12_frame import Nv12Frame, Nv12SbsFrame
from platform.jetson.cuda_runtime import CudaEvent, CudaStream
from platform.jetson.nvbuf_egl_cuda import Nv12CudaPitchPlanes
from platform.jetson.trt import YOLO26PoseTRTRunner, YOLO26PoseStereoTRTRunner

logger = logging.getLogger(__name__)

_infer_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="route2_trt")


@dataclass
class _PreparedEyeInput:
    ow: int
    oh: int
    geo: Optional[LetterboxGeometry]
    vic_unpad_ms: float = 0.0
    vic_place_ms: float = 0.0
    egl_ms: float = 0.0
    nv12_planes: Optional[Nv12CudaPitchPlanes] = None
    bgr_canvas: Optional[np.ndarray] = None
    trt_prefed: bool = False


@dataclass
class _TrtEyeOutput:
    raw: np.ndarray
    pre_ms: float = 0.0
    infer_ms: float = 0.0


@dataclass
class _PrepOverlapResult:
    egl_session: Any
    full_w: int
    full_h: int
    eye_w: int
    capture_seq: int


@dataclass
class _OverlapState:
    inflight_future: Future[_PrepOverlapResult] | None = None
    inflight_sbs: Any | None = None
    next_slot: int = 0


@dataclass
class _VicBgrOverlapState:
    """vic_bgr：prep(N+1) VIC→BGR ∥ infer(N) letterbox+TRT。"""

    inflight_future: Future[Any] | None = None
    inflight_sbs: Any | None = None


@dataclass
class EyePoseResult:
    poses: list[dict] = field(default_factory=list)
    vic_unpad_ms: float = 0.0
    vic_place_ms: float = 0.0
    egl_ms: float = 0.0
    pre_ms: float = 0.0
    infer_ms: float = 0.0
    post_ms: float = 0.0

    @property
    def vic_ms(self) -> float:
        return self.vic_unpad_ms + self.vic_place_ms + self.egl_ms

    @property
    def eye_total_ms(self) -> float:
        return self.vic_ms + self.pre_ms + self.infer_ms + self.post_ms


@dataclass
class StereoInferResult:
    frame_id: int
    left: EyePoseResult
    right: EyePoseResult
    capture_ms: float = 0.0
    pull_ms: float = 0.0
    vic_prep_ms: float = 0.0
    capture_seq: int = -1
    infer_parallel_ms: float = 0.0
    preview_bgr: np.ndarray | None = None
    preview_ms: float = 0.0
    par_drain_ms: float = 0.0
    par_wait_ms: float = 0.0
    par_unmap_ms: float = 0.0
    par_trt_enqueue_ms: float = 0.0
    par_trt_d2h_sync_ms: float = 0.0

    @property
    def pipeline_ms(self) -> float:
        return self.capture_ms + self.infer_parallel_ms

    def par_breakdown(self) -> "InferParBreakdown":
        return InferParBreakdown.from_stereo(self)


@dataclass(frozen=True)
class InferParBreakdown:
    """par 墙钟分项（并行段取 max，不含 cap / 三角化）。"""

    egl_ms: float
    lb_ms: float
    trt_ms: float
    post_ms: float
    drain_ms: float = 0.0
    wait_ms: float = 0.0
    unmap_ms: float = 0.0
    trt_enqueue_ms: float = 0.0
    trt_d2h_sync_ms: float = 0.0

    @classmethod
    def from_stereo(cls, infer: StereoInferResult) -> InferParBreakdown:
        left, right = infer.left, infer.right
        common = dict(
            drain_ms=float(infer.par_drain_ms),
            wait_ms=float(infer.par_wait_ms),
            unmap_ms=float(infer.par_unmap_ms),
            trt_enqueue_ms=float(infer.par_trt_enqueue_ms),
            trt_d2h_sync_ms=float(infer.par_trt_d2h_sync_ms),
        )
        if infer.vic_prep_ms > 0.0:
            return cls(
                egl_ms=max(float(left.egl_ms), float(right.egl_ms)),
                lb_ms=max(float(left.pre_ms), float(right.pre_ms)),
                trt_ms=max(float(left.infer_ms), float(right.infer_ms)),
                post_ms=max(float(left.post_ms), float(right.post_ms)),
                **common,
            )
        return cls(
            egl_ms=float(left.egl_ms),
            lb_ms=max(float(left.vic_unpad_ms), float(right.vic_unpad_ms)),
            trt_ms=max(float(left.infer_ms), float(right.infer_ms)),
            post_ms=max(float(left.post_ms), float(right.post_ms)),
            **common,
        )

    @property
    def accounted_ms(self) -> float:
        return (
            self.egl_ms
            + self.lb_ms
            + self.trt_ms
            + self.post_ms
            + self.drain_ms
            + self.wait_ms
            + self.unmap_ms
        )

    def format_line(self) -> str:
        base = (
            f"egl={self.egl_ms:.1f} lb={self.lb_ms:.1f} "
            f"trt={self.trt_ms:.1f} post={self.post_ms:.1f}"
        )
        extra = (
            f" drain={self.drain_ms:.1f} wait={self.wait_ms:.1f} "
            f"unmap={self.unmap_ms:.1f} trt_eq={self.trt_enqueue_ms:.1f} "
            f"trt_sync={self.trt_d2h_sync_ms:.1f}"
        )
        return base + extra


class StereoInferPipeline:
    """默认 GPU letterbox + EGL-CUDA + 双眼 TRT 并行（对齐 YOLOs-CPP 官方思路）。"""

    def __init__(
        self,
        engine_path: Path | str = DEFAULT_YOLO26_POSE_ENGINE,
        *,
        conf_threshold: float = 0.25,
        use_egl_direct: bool = True,
        gpu_letterbox: bool = True,
        infer_prep: str = "vic_bgr",
        trt_only: bool = False,
        stereo_letterbox_fused: bool = True,
        eye_rotate: str = "none",
    ) -> None:
        if not gst_nvmm_available():
            raise RuntimeError("需要 GStreamer NVMM (gi + libnvbufsurface)")

        self.conf_threshold = conf_threshold
        self.trt_only = bool(trt_only)
        self.use_egl_direct = use_egl_direct
        self._engine_path = Path(engine_path)
        from core.geometry.eye_mount_rotate import parse_eye_mount_rotate

        self.eye_rotate = parse_eye_mount_rotate(eye_rotate)
        prep_key = str(infer_prep).strip().lower()
        if prep_key not in ("gpu", "vic_bgr"):
            raise ValueError(f"infer_prep 须 gpu|vic_bgr，收到 {infer_prep!r}")
        self.infer_prep = prep_key
        self._use_vic_bgr = self.infer_prep == "vic_bgr" and not self.trt_only
        if self._use_vic_bgr:
            gpu_letterbox = False
        self.gpu_letterbox = bool(gpu_letterbox) and use_egl_direct and not self.trt_only
        self.stereo_letterbox_fused = bool(stereo_letterbox_fused) and self.gpu_letterbox
        self._use_vic_letterbox = (
            use_egl_direct
            and not self.gpu_letterbox
            and not self.trt_only
            and not self._use_vic_bgr
        )
        if self._use_vic_letterbox:
            raise RuntimeError(
                "VIC letterbox 路径已移除；请用 infer_prep=vic_bgr 或 gpu（默认 vic_bgr）"
            )
        if self._use_vic_bgr:
            from platform.jetson.nvbuf_vic_scale import nvbuf_vic_scale_available

            if not nvbuf_vic_scale_available():
                raise RuntimeError("infer_prep=vic_bgr 需要 NvBuf VIC")

        if self.trt_only:
            if not cuda_nv12_rgb_available():
                raise RuntimeError(
                    "TRT-only 需要 route2_nv12_rgb.so: "
                    "bash scripts/route2/build_cuda_nv12_rgb.sh"
                )
        elif self.gpu_letterbox:
            if not gpu_letterbox_cuda_available():
                raise RuntimeError(
                    "GPU letterbox 需要 route2_nv12_rgb.so（含 letterbox 符号）: "
                    "bash scripts/route2/build_cuda_nv12_rgb.sh"
                )
            if not gpu_crop_letterbox_cuda_available():
                raise RuntimeError(
                    "GPU SBS 裁半需要 crop letterbox 符号: "
                    "bash scripts/route2/build_cuda_nv12_rgb.sh"
                )
            if self.stereo_letterbox_fused and not gpu_stereo_sbs_letterbox_cuda_available():
                logger.warning(
                    "stereo fused letterbox 不可用，回退双 kernel；"
                    "请运行 bash scripts/route2/build_cuda_nv12_rgb.sh"
                )
                self.stereo_letterbox_fused = False
        elif self._use_vic_letterbox:
            from platform.jetson.vic import route2_nvbuf_vic_available

            if not route2_nvbuf_vic_available():
                raise RuntimeError("需要 NvBufSurfTransform VIC")

        if use_egl_direct and not cuda_nv12_rgb_available():
            raise RuntimeError(
                "EGL 直绑需要 route2_nv12_rgb.so: bash scripts/route2/build_cuda_nv12_rgb.sh"
            )

        self._stereo_trt: YOLO26PoseStereoTRTRunner | None = None
        b2_path = resolve_yolo26_stereo_engine(Path(engine_path))
        if b2_path is not None and self.gpu_letterbox and not self.trt_only:
            self._stereo_trt = YOLO26PoseStereoTRTRunner(b2_path)
            self._left_trt = self._stereo_trt
            self._right_trt = None
        else:
            self._left_trt = YOLO26PoseTRTRunner(engine_path)
            self._right_trt = YOLO26PoseTRTRunner(engine_path)
        self._left_gpu: Any | None = None
        self._right_gpu: Any | None = None
        self._left_vic: Any | None = None
        self._right_vic: Any | None = None
        self._vic_sbs_prep: Any | None = None
        self._overlap: _OverlapState | None = None
        self._vic_bgr_overlap: _VicBgrOverlapState | None = None
        self._overlap_prep_stream: CudaStream | None = None
        self._overlap_prep_aux_stream: CudaStream | None = None
        self._overlap_egl_event: CudaEvent | None = None
        self._vic_bgr: Any | None = None
        self._vic_bgr_eye_w = 0
        self._vic_bgr_eye_h = 0
        self._vic_bgr_eye_rotate = "none"

        if self.gpu_letterbox:
            from platform.jetson.route2_gpu_preprocess import Route2GpuPreprocessor

            self._left_gpu = Route2GpuPreprocessor()
            self._right_gpu = Route2GpuPreprocessor()
        elif self._use_vic_letterbox:
            from platform.jetson.vic import Route2VicPreprocessor

            self._left_vic = Route2VicPreprocessor()
            self._right_vic = Route2VicPreprocessor()
            self._left_vic.warm_up(1920, 1080)
            self._right_vic.warm_up(1920, 1080)

        self._frame_id = 0
        if self.trt_only:
            mode = "vic_capture_trt_only"
        elif self._use_vic_bgr:
            mode = "nvdec_vic_bgr_cpu"
        elif self.gpu_letterbox:
            mode = "egl_cuda_gpu_letterbox"
        elif self.use_egl_direct:
            mode = "egl_cuda_vic_letterbox"
        else:
            mode = "bgr640_bridge"
        logger.info(
            "StereoInferPipeline mode=%s trt=%s prep=%s",
            mode,
            f"batch={self._stereo_trt.batch_size}" if self._stereo_trt else "dual_engine",
            "capture_vic"
            if self.trt_only
            else (
                "vic_bgr_cpu"
                if self._use_vic_bgr
                else ("gpu_cuda" if self.gpu_letterbox else "vic_serial")
            ),
        )

    def _ensure_overlap_streams(self) -> tuple[CudaStream, CudaStream]:
        if self._overlap_prep_stream is None:
            self._overlap_prep_stream = CudaStream()
        if self._overlap_prep_aux_stream is None:
            self._overlap_prep_aux_stream = CudaStream()
        return self._overlap_prep_stream, self._overlap_prep_aux_stream

    def _ensure_overlap_egl_event(self) -> CudaEvent:
        if self._overlap_egl_event is None:
            self._overlap_egl_event = CudaEvent()
        return self._overlap_egl_event

    def _mark_overlap_egl_ready(self, record_stream: CudaStream) -> None:
        """EGL map 完成后 record；主线程 trt stream wait 后 letterbox+graph。"""
        self._ensure_overlap_egl_event().record(record_stream)

    def _wait_overlap_egl_ready(self) -> None:
        st = self._stereo_trt
        assert st is not None
        ev = self._overlap_egl_event
        if ev is None:
            st.stream.sync()
            return
        ev.stream_wait(st.stream)

    def _prepare_one_eye_gpu(
        self,
        eye: Nv12Frame,
        gpu: Any,
        trt: YOLO26PoseTRTRunner,
        *,
        full_w: int | None = None,
        full_h: int | None = None,
        crop_x: int = 0,
        crop_y: int = 0,
    ) -> _PreparedEyeInput:
        if eye.dmabuf_fd is None:
            raise ValueError("Nv12Frame 需要 dmabuf_fd")
        fw = int(full_w if full_w is not None else eye.width)
        fh = int(full_h if full_h is not None else eye.height)
        cw, ch = eye.width, eye.height
        geo, prof = gpu.feed_trt_from_fd(
            trt,
            int(eye.dmabuf_fd),
            fw,
            fh,
            crop_x=int(crop_x),
            crop_y=int(crop_y),
            crop_w=int(cw),
            crop_h=int(ch),
        )
        return _PreparedEyeInput(
            ow=cw,
            oh=ch,
            geo=geo,
            vic_unpad_ms=prof.letterbox_ms,
            egl_ms=prof.egl_bind_ms,
            trt_prefed=True,
        )

    def _prepare_one_eye_vic(
        self,
        eye: Nv12Frame,
        vic: Any,
        *,
        full_bgr: Optional[np.ndarray] = None,
    ) -> _PreparedEyeInput:
        if eye.dmabuf_fd is None:
            raise ValueError("Nv12Frame 需要 dmabuf_fd")
        ow, oh = eye.width, eye.height
        if self.use_egl_direct:
            planes, vprof = vic.prepare_nv12_cuda_from_fd(eye.dmabuf_fd, ow, oh)
            return _PreparedEyeInput(
                ow=ow,
                oh=oh,
                geo=vic.last_geometry,
                vic_unpad_ms=vprof.unpad_ms,
                vic_place_ms=vprof.place_ms,
                egl_ms=vprof.egl_bind_ms,
                nv12_planes=planes,
            )
        from platform.jetson.letterbox_util import letterbox_bgr_cpu

        t_v = time.perf_counter()
        if full_bgr is not None:
            canvas, geo = letterbox_bgr_cpu(full_bgr, 640, 640)
            vic._last_geometry = geo
        else:
            canvas = vic.prepare_bgr_from_fd(eye.dmabuf_fd, ow, oh)
            geo = vic.last_geometry
        vic_ms = (time.perf_counter() - t_v) * 1000.0
        return _PreparedEyeInput(
            ow=ow,
            oh=oh,
            geo=geo,
            vic_unpad_ms=vic_ms,
            bgr_canvas=canvas,
        )

    def _prepare_one_eye(
        self,
        eye: Nv12Frame,
        side: str,
        *,
        full_bgr: Optional[np.ndarray] = None,
        full_w: int | None = None,
        full_h: int | None = None,
        crop_x: int = 0,
        crop_y: int = 0,
    ) -> _PreparedEyeInput:
        if self.gpu_letterbox:
            gpu = self._left_gpu if side == "left" else self._right_gpu
            trt = self._left_trt if side == "left" else self._right_trt
            assert gpu is not None
            return self._prepare_one_eye_gpu(
                eye,
                gpu,
                trt,
                full_w=full_w,
                full_h=full_h,
                crop_x=crop_x,
                crop_y=crop_y,
            )
        vic = self._left_vic if side == "left" else self._right_vic
        assert vic is not None
        return self._prepare_one_eye_vic(eye, vic, full_bgr=full_bgr)

    def _run_trt(
        self,
        trt: YOLO26PoseTRTRunner,
        prep: _PreparedEyeInput,
    ) -> _TrtEyeOutput:
        if prep.trt_prefed:
            raw = trt.infer_trt_only()
        elif prep.nv12_planes is not None:
            trt.feed_nv12_planes_to_input(prep.nv12_planes)
            raw = trt.infer_trt_only()
        else:
            assert prep.bgr_canvas is not None
            raw = trt.infer_bgr_letterboxed(prep.bgr_canvas)
        return _TrtEyeOutput(
            raw=raw,
            pre_ms=trt.last_preprocess_ms,
            infer_ms=trt.last_infer_ms,
        )

    def _post_one_eye(
        self,
        prep: _PreparedEyeInput,
        trt_out: _TrtEyeOutput,
    ) -> EyePoseResult:
        t_post = time.perf_counter()
        if prep.geo is None:
            raise RuntimeError("letterbox geometry 缺失")
        poses = postprocess_v26(
            trt_out.raw,
            (prep.ow, prep.oh),
            letterbox_geometry=prep.geo,
            conf_threshold=self.conf_threshold,
        )
        post_ms = (time.perf_counter() - t_post) * 1000.0
        return EyePoseResult(
            poses=poses,
            vic_unpad_ms=prep.vic_unpad_ms,
            vic_place_ms=prep.vic_place_ms,
            egl_ms=prep.egl_ms,
            pre_ms=trt_out.pre_ms,
            infer_ms=trt_out.infer_ms,
            post_ms=post_ms,
        )

    def _prepare_sbs_gpu_crop_stereo(
        self,
        left_eye: Nv12Frame,
        *,
        full_w: int,
        full_h: int,
        eye_w: int,
        with_gpu_preview: bool = False,
    ) -> tuple[_PreparedEyeInput, _PreparedEyeInput, np.ndarray | None, float]:
        from core.display.gpu_preview import get_gpu_sbs_preview
        from platform.jetson.sbs_gpu_egl_session import SbsGpuEglSession

        if left_eye.dmabuf_fd is None:
            raise ValueError("Nv12Frame 需要 dmabuf_fd")
        assert self._left_gpu is not None and self._right_gpu is not None

        st = self._stereo_trt
        use_fused = (
            st is not None
            and self.stereo_letterbox_fused
            and gpu_stereo_sbs_letterbox_cuda_available()
        )
        lb_stream = (
            st.stream.handle
            if st is not None
            else self._left_trt.stream.handle
        )
        preview_bgr: np.ndarray | None = None
        preview_ms = 0.0
        egl_ms = 0.0
        with SbsGpuEglSession.open(
            int(left_eye.dmabuf_fd),
            full_w,
            full_h,
            stream_handle=lb_stream,
        ) as session:
            egl_ms = session.egl_bind_ms
            lb_kw = dict(crop_y=0, crop_h=full_h)
            if use_fused:
                geo, lb_ms = session.letterbox_stereo_crop_to_trt(
                    st,
                    self._left_gpu,
                    eye_w=eye_w,
                    stream_handle=lb_stream,
                    dst_ptr0=st.input_device_ptr_at(0),
                    dst_ptr1=st.input_device_ptr_at(1),
                    **lb_kw,
                )
                geo_l, geo_r = geo, geo
                lb_l, lb_r = lb_ms, lb_ms
                if st is not None:
                    st.stream.sync()
            elif st is not None:
                stream_l = st.stream.handle
                stream_r = st._letterbox_aux_stream.handle
                fut_l = _infer_pool.submit(
                    session.letterbox_crop_to_trt,
                    st,
                    self._left_gpu,
                    crop_x=0,
                    crop_w=eye_w,
                    stream_handle=stream_l,
                    dst_ptr=st.input_device_ptr_at(0),
                    **lb_kw,
                )
                fut_r = _infer_pool.submit(
                    session.letterbox_crop_to_trt,
                    st,
                    self._right_gpu,
                    crop_x=eye_w,
                    crop_w=eye_w,
                    stream_handle=stream_r,
                    dst_ptr=st.input_device_ptr_at(1),
                    **lb_kw,
                )
                geo_l, lb_l = fut_l.result()
                geo_r, lb_r = fut_r.result()
                st.stream.sync()
                st._letterbox_aux_stream.sync()
            else:
                assert self._right_trt is not None
                stream_l = self._left_trt.stream.handle
                stream_r = self._right_trt.stream.handle
                fut_l = _infer_pool.submit(
                    session.letterbox_crop_to_trt,
                    self._left_trt,
                    self._left_gpu,
                    crop_x=0,
                    crop_w=eye_w,
                    stream_handle=stream_l,
                    **lb_kw,
                )
                fut_r = _infer_pool.submit(
                    session.letterbox_crop_to_trt,
                    self._right_trt,
                    self._right_gpu,
                    crop_x=eye_w,
                    crop_w=eye_w,
                    stream_handle=stream_r,
                    **lb_kw,
                )
                geo_l, lb_l = fut_l.result()
                geo_r, lb_r = fut_r.result()
                self._left_trt.stream.sync()
                self._right_trt.stream.sync()
            if with_gpu_preview:
                out = session.preview_left_bgr(
                    get_gpu_sbs_preview(),
                    eye_w=eye_w,
                )
                preview_bgr = out.bgr
                preview_ms = out.preview_ms

        prep_l = _PreparedEyeInput(
            ow=eye_w,
            oh=full_h,
            geo=geo_l,
            vic_unpad_ms=lb_l,
            egl_ms=egl_ms,
            trt_prefed=True,
        )
        prep_r = _PreparedEyeInput(
            ow=eye_w,
            oh=full_h,
            geo=geo_r,
            vic_unpad_ms=lb_r,
            egl_ms=0.0,
            trt_prefed=True,
        )
        self._left_gpu._last_geometry = geo_l
        self._right_gpu._last_geometry = geo_r
        if st is not None:
            if use_fused:
                st.last_preprocess_ms = egl_ms + lb_l
            else:
                st.last_preprocess_ms = egl_ms + lb_l + lb_r
        else:
            self._left_trt.last_preprocess_ms = egl_ms + lb_l
            assert self._right_trt is not None
            self._right_trt.last_preprocess_ms = lb_r
        return prep_l, prep_r, preview_bgr, preview_ms

    def _ensure_vic_sbs_prep(
        self,
        *,
        full_w: int,
        full_h: int,
        eye_w: int,
    ) -> Any:
        from platform.jetson.sbs_vic_capture_prep import SbsVicCapturePreparer

        prep = self._vic_sbs_prep
        if prep is None:
            prep = SbsVicCapturePreparer(
                full_width=int(full_w),
                full_height=int(full_h),
                eye_width=int(eye_w),
                eye_height=int(full_h),
                pool_depth=2,
            )
            self._vic_sbs_prep = prep
        return prep

    def _prepare_both_eyes(
        self,
        left_eye: Nv12Frame,
        right_eye: Nv12Frame,
        *,
        full_w: int | None = None,
        full_h: int | None = None,
        eye_w: int | None = None,
        sbs_gpu_crop: bool = False,
    ) -> tuple[_PreparedEyeInput, _PreparedEyeInput]:
        if sbs_gpu_crop and self.gpu_letterbox:
            raise RuntimeError("_prepare_both_eyes 不应在 sbs_gpu_crop 模式调用")
        if self.gpu_letterbox:
            prep_l = self._prepare_one_eye(left_eye, "left")
            prep_r = self._prepare_one_eye(right_eye, "right")
            return prep_l, prep_r
        prep_l = self._prepare_one_eye(left_eye, "left")
        prep_r = self._prepare_one_eye(right_eye, "right")
        return prep_l, prep_r

    def _infer_eyes(
        self,
        left_eye: Nv12Frame,
        right_eye: Nv12Frame,
        *,
        parallel: bool = True,
        full_w: int | None = None,
        full_h: int | None = None,
        eye_w: int | None = None,
        sbs_gpu_crop: bool = False,
        with_gpu_preview: bool = False,
    ) -> tuple[EyePoseResult, EyePoseResult, float, np.ndarray | None, float]:
        t_block = time.perf_counter()
        preview_bgr: np.ndarray | None = None
        preview_ms = 0.0
        if sbs_gpu_crop and self.gpu_letterbox:
            fw = int(full_w) if full_w else int_attr(left_eye, "width", default=0)
            fh = int(full_h) if full_h else int_attr(left_eye, "height", default=0)
            ew = int(eye_w) if eye_w else int_attr(left_eye, "width", default=1920)
            prep_l, prep_r, preview_bgr, preview_ms = self._prepare_sbs_gpu_crop_stereo(
                left_eye,
                full_w=fw,
                full_h=fh,
                eye_w=ew,
                with_gpu_preview=with_gpu_preview,
            )
        else:
            prep_kw = dict(
                full_w=full_w,
                full_h=full_h,
                eye_w=eye_w,
                sbs_gpu_crop=False,
            )
            prep_l, prep_r = self._prepare_both_eyes(
                left_eye, right_eye, **prep_kw
            )

        if self._stereo_trt is not None and prep_l.trt_prefed and prep_r.trt_prefed:
            raw_b = self._stereo_trt.infer_trt_only()
            infer_ms = float(self._stereo_trt.last_infer_ms)
            trt_l = _TrtEyeOutput(raw=raw_b[0], infer_ms=infer_ms)
            trt_r = _TrtEyeOutput(raw=raw_b[1], infer_ms=infer_ms)
            if parallel:
                fut_pl = _infer_pool.submit(self._post_one_eye, prep_l, trt_l)
                fut_pr = _infer_pool.submit(self._post_one_eye, prep_r, trt_r)
                left_res = fut_pl.result()
                right_res = fut_pr.result()
            else:
                left_res = self._post_one_eye(prep_l, trt_l)
                right_res = self._post_one_eye(prep_r, trt_r)
        elif parallel:
            assert self._right_trt is not None
            fut_l = _infer_pool.submit(self._run_trt, self._left_trt, prep_l)
            fut_r = _infer_pool.submit(self._run_trt, self._right_trt, prep_r)
            trt_l = fut_l.result()
            trt_r = fut_r.result()
            left_res = _infer_pool.submit(
                self._post_one_eye, prep_l, trt_l
            ).result()
            right_res = _infer_pool.submit(
                self._post_one_eye, prep_r, trt_r
            ).result()
        else:
            assert self._right_trt is not None
            trt_l = self._run_trt(self._left_trt, prep_l)
            trt_r = self._run_trt(self._right_trt, prep_r)
            left_res = self._post_one_eye(prep_l, trt_l)
            right_res = self._post_one_eye(prep_r, trt_r)

        infer_parallel_ms = (time.perf_counter() - t_block) * 1000.0
        return left_res, right_res, infer_parallel_ms, preview_bgr, preview_ms

    def _sbs_crop_eye_from_full(self, full: Nv12Frame, eye_w: int) -> Nv12Frame:
        return Nv12Frame(
            y=full.y,
            uv=full.uv,
            width=eye_w,
            height=full.height,
            dmabuf_fd=full.dmabuf_fd,
            pitch=full.pitch,
            zero_copy=full.zero_copy,
            _keepalive=full._keepalive,
        )

    def _submit_overlap_egl(
        self,
        sbs: Any,
        *,
        full_w: int,
        full_h: int,
        eye_w: int,
    ) -> Future[_PrepOverlapResult]:
        from platform.jetson.sbs_gpu_egl_session import SbsGpuEglSession

        full = sbs.full
        assert full is not None and full.dmabuf_fd is not None
        capture_seq = int_attr(sbs, "capture_seq", default=-1)
        pipeline = self

        def _job() -> _PrepOverlapResult:
            session: SbsGpuEglSession | None = None
            try:
                prep_stream, _ = pipeline._ensure_overlap_streams()
                session = SbsGpuEglSession.open(
                    int(full.dmabuf_fd),
                    int(full_w),
                    int(full_h),
                    stream_handle=prep_stream.handle,
                )
                pipeline._mark_overlap_egl_ready(prep_stream)
                return _PrepOverlapResult(
                    egl_session=session,
                    full_w=int(full_w),
                    full_h=int(full_h),
                    eye_w=int(eye_w),
                    capture_seq=capture_seq,
                )
            except Exception:
                if session is not None:
                    session.close()
                raise

        return _infer_pool.submit(_job)

    def _stereo_trt_split_ms(self) -> tuple[float, float, float]:
        st = self._stereo_trt
        if st is None:
            return 0.0, 0.0, 0.0
        total = float(st.last_infer_ms)
        enqueue = float(getattr(st, "last_infer_enqueue_ms", 0.0))
        d2h_sync = float(getattr(st, "last_infer_d2h_sync_ms", 0.0))
        return total, enqueue, d2h_sync

    def _drain_overlap_egl(
        self,
        overlap: _OverlapState,
    ) -> tuple[Any, int, int, int, int, float]:
        if overlap.inflight_future is None:
            raise RuntimeError("overlap EGL future 缺失")
        t0 = time.perf_counter()
        fut = overlap.inflight_future
        overlap.inflight_future = None
        try:
            prep_result = fut.result()
        except Exception:
            if overlap.inflight_sbs is not None:
                try:
                    overlap.inflight_sbs.release()
                except Exception:
                    pass
                overlap.inflight_sbs = None
            raise
        if overlap.inflight_sbs is not None:
            try:
                overlap.inflight_sbs.release()
            except Exception:
                pass
            overlap.inflight_sbs = None
        drain_ms = (time.perf_counter() - t0) * 1000.0
        return (
            prep_result.egl_session,
            prep_result.full_w,
            prep_result.full_h,
            prep_result.eye_w,
            prep_result.capture_seq,
            drain_ms,
        )

    def _close_overlap_egl_session(self, session: Any | None) -> None:
        if session is None:
            return
        try:
            session.close()
        except Exception:
            pass

    def _letterbox_from_egl_session(
        self,
        session: Any,
        *,
        full_h: int,
        eye_w: int,
    ) -> tuple[_PreparedEyeInput, _PreparedEyeInput]:
        st = self._stereo_trt
        assert st is not None and self._left_gpu is not None and self._right_gpu is not None
        lb_stream = st.stream.handle
        use_fused = (
            self.stereo_letterbox_fused
            and gpu_stereo_sbs_letterbox_cuda_available()
        )
        lb_kw = dict(crop_y=0, crop_h=int(full_h))
        egl_ms = float(getattr(session, "egl_bind_ms", 0.0))
        if use_fused:
            geo, lb_ms = session.letterbox_stereo_crop_to_trt(
                st,
                self._left_gpu,
                eye_w=int(eye_w),
                stream_handle=lb_stream,
                dst_ptr0=st.input_device_ptr_at(0),
                dst_ptr1=st.input_device_ptr_at(1),
                **lb_kw,
            )
            geo_l, geo_r = geo, geo
            lb_l, lb_r = lb_ms, lb_ms
        else:
            stream_r = st._letterbox_aux_stream.handle
            fut_l = _infer_pool.submit(
                session.letterbox_crop_to_trt,
                st,
                self._left_gpu,
                crop_x=0,
                crop_w=int(eye_w),
                stream_handle=lb_stream,
                dst_ptr=st.input_device_ptr_at(0),
                **lb_kw,
            )
            fut_r = _infer_pool.submit(
                session.letterbox_crop_to_trt,
                st,
                self._right_gpu,
                crop_x=int(eye_w),
                crop_w=int(eye_w),
                stream_handle=stream_r,
                dst_ptr=st.input_device_ptr_at(1),
                **lb_kw,
            )
            geo_l, lb_l = fut_l.result()
            geo_r, lb_r = fut_r.result()
            st._letterbox_aux_stream.sync()
        st.last_preprocess_ms = egl_ms + lb_l + (0.0 if use_fused else lb_r)
        prep_l = _PreparedEyeInput(
            ow=int(eye_w),
            oh=int(full_h),
            geo=geo_l,
            vic_unpad_ms=lb_l,
            egl_ms=egl_ms,
            trt_prefed=True,
        )
        prep_r = _PreparedEyeInput(
            ow=int(eye_w),
            oh=int(full_h),
            geo=geo_r,
            vic_unpad_ms=lb_r,
            egl_ms=0.0,
            trt_prefed=True,
        )
        self._left_gpu._last_geometry = geo_l
        self._right_gpu._last_geometry = geo_r
        return prep_l, prep_r

    def reset_vic_bgr_overlap_state(self) -> None:
        overlap = self._vic_bgr_overlap
        if overlap is None:
            return
        if overlap.inflight_future is not None:
            try:
                overlap.inflight_future.cancel()
            except Exception:
                pass
            try:
                overlap.inflight_future.result(timeout=0.05)
            except Exception:
                pass
            overlap.inflight_future = None
        if overlap.inflight_sbs is not None:
            try:
                overlap.inflight_sbs.release()
            except Exception:
                pass
            overlap.inflight_sbs = None

    def reset_overlap_state(self) -> None:
        """丢弃 inflight prep（infer 异常后避免后续帧连锁 int(None)/result 重抛）。"""
        self.reset_vic_bgr_overlap_state()
        overlap = self._overlap
        if overlap is None:
            return
        if overlap.inflight_future is not None:
            try:
                overlap.inflight_future.cancel()
            except Exception:
                pass
            try:
                prep_result = overlap.inflight_future.result(timeout=0.05)
                self._close_overlap_egl_session(prep_result.egl_session)
            except Exception:
                pass
            overlap.inflight_future = None
        if overlap.inflight_sbs is not None:
            try:
                overlap.inflight_sbs.release()
            except Exception:
                pass
            overlap.inflight_sbs = None

    def _infer_stereo_batch_post(
        self,
        prep_l: _PreparedEyeInput,
        prep_r: _PreparedEyeInput,
        *,
        parallel: bool,
    ) -> tuple[EyePoseResult, EyePoseResult]:
        assert self._stereo_trt is not None
        raw_b = self._stereo_trt.infer_trt_only()
        infer_ms = float(self._stereo_trt.last_infer_ms)
        trt_l = _TrtEyeOutput(raw=raw_b[0], infer_ms=infer_ms)
        trt_r = _TrtEyeOutput(raw=raw_b[1], infer_ms=infer_ms)
        if parallel:
            fut_pl = _infer_pool.submit(self._post_one_eye, prep_l, trt_l)
            fut_pr = _infer_pool.submit(self._post_one_eye, prep_r, trt_r)
            return fut_pl.result(), fut_pr.result()
        return self._post_one_eye(prep_l, trt_l), self._post_one_eye(prep_r, trt_r)

    def _run_stereo_egl_letterbox_trt_post(
        self,
        egl_session: Any,
        *,
        full_h: int,
        eye_w: int,
        parallel: bool,
        egl_overlap: bool,
    ) -> tuple[EyePoseResult, EyePoseResult, float, float]:
        wait_ms = 0.0
        unmap_ms = 0.0
        try:
            if egl_overlap:
                t_wait = time.perf_counter()
                self._wait_overlap_egl_ready()
                wait_ms = (time.perf_counter() - t_wait) * 1000.0
            prep_l, prep_r = self._letterbox_from_egl_session(
                egl_session,
                full_h=full_h,
                eye_w=eye_w,
            )
            left_res, right_res = self._infer_stereo_batch_post(
                prep_l,
                prep_r,
                parallel=parallel,
            )
        finally:
            t_unmap = time.perf_counter()
            self._close_overlap_egl_session(egl_session)
            unmap_ms = (time.perf_counter() - t_unmap) * 1000.0
        return left_res, right_res, wait_ms, unmap_ms

    def _process_sbs_overlap(
        self,
        sbs: Any,
        capture_ms: float,
        *,
        parallel_eyes: bool = True,
        release_sbs: bool = True,
        prefetch_sbs: Any | None = None,
    ) -> StereoInferResult | None:
        self._frame_id += 1
        full = sbs.full
        if full is None or full.dmabuf_fd is None:
            if release_sbs:
                sbs.release()
            return None

        full_w = int_attr(sbs, "width", default=0)
        full_h = int_attr(sbs, "height", default=0)
        eye_w = full_w // 2 if full_w > 0 else 1920
        overlap = self._overlap
        if overlap is None:
            overlap = _OverlapState()
            self._overlap = overlap

        t_block = time.perf_counter()
        result_capture_seq = int_attr(sbs, "capture_seq", default=-1)
        par_drain_ms = 0.0
        par_wait_ms = 0.0
        par_unmap_ms = 0.0
        try:
            if overlap.inflight_future is not None:
                egl_session, _fw, fh, ew, result_capture_seq, par_drain_ms = (
                    self._drain_overlap_egl(overlap)
                )
                left_res, right_res, par_wait_ms, par_unmap_ms = (
                    self._run_stereo_egl_letterbox_trt_post(
                        egl_session,
                        full_h=fh,
                        eye_w=ew,
                        parallel=parallel_eyes,
                        egl_overlap=True,
                    )
                )
            else:
                left_eye = self._sbs_crop_eye_from_full(full, eye_w)
                prep_l, prep_r, _, _ = self._prepare_sbs_gpu_crop_stereo(
                    left_eye,
                    full_w=full_w,
                    full_h=full_h,
                    eye_w=eye_w,
                )
                left_res, right_res = self._infer_stereo_batch_post(
                    prep_l,
                    prep_r,
                    parallel=parallel_eyes,
                )

            if prefetch_sbs is not None:
                pf_full = getattr(prefetch_sbs, "full", None)
                if pf_full is not None and pf_full.dmabuf_fd is not None:
                    pf_w = int_attr(prefetch_sbs, "width", default=full_w)
                    pf_h = int_attr(prefetch_sbs, "height", default=full_h)
                    pf_eye_w = pf_w // 2 if pf_w > 0 else eye_w
                    overlap.inflight_sbs = prefetch_sbs
                    overlap.inflight_future = self._submit_overlap_egl(
                        prefetch_sbs,
                        full_w=pf_w,
                        full_h=pf_h,
                        eye_w=pf_eye_w,
                    )
        except Exception:
            self.reset_overlap_state()
            raise
        finally:
            if release_sbs:
                sbs.release()

        infer_parallel_ms = (time.perf_counter() - t_block) * 1000.0
        _, par_trt_enqueue_ms, par_trt_d2h_sync_ms = self._stereo_trt_split_ms()
        return StereoInferResult(
            frame_id=self._frame_id,
            left=left_res,
            right=right_res,
            capture_ms=capture_ms,
            capture_seq=result_capture_seq,
            infer_parallel_ms=infer_parallel_ms,
            par_drain_ms=par_drain_ms,
            par_wait_ms=par_wait_ms,
            par_unmap_ms=par_unmap_ms,
            par_trt_enqueue_ms=par_trt_enqueue_ms,
            par_trt_d2h_sync_ms=par_trt_d2h_sync_ms,
        )

    def _ensure_vic_bgr(self, *, eye_w: int, eye_h: int) -> Any:
        from core.infer.vic_bgr_stereo import VicBgrStereoInfer

        rot = str(self.eye_rotate)
        if (
            self._vic_bgr is None
            or self._vic_bgr_eye_w != int(eye_w)
            or self._vic_bgr_eye_h != int(eye_h)
            or self._vic_bgr_eye_rotate != rot
        ):
            if self._vic_bgr is not None:
                self._vic_bgr.close()
            self._vic_bgr = VicBgrStereoInfer(
                self._engine_path,
                conf_threshold=self.conf_threshold,
                eye_w=int(eye_w),
                eye_h=int(eye_h),
                eye_rotate=rot,
            )
            self._vic_bgr_eye_w = int(eye_w)
            self._vic_bgr_eye_h = int(eye_h)
            self._vic_bgr_eye_rotate = rot
        return self._vic_bgr

    def _process_sbs_vic_bgr(
        self,
        sbs: Any,
        capture_ms: float,
        *,
        parallel_eyes: bool = True,
        release_sbs: bool = True,
        with_gpu_preview: bool = False,
    ) -> StereoInferResult | None:
        full = sbs.full
        if full is None or full.dmabuf_fd is None:
            if release_sbs:
                sbs.release()
            return None
        full_h = int_attr(sbs, "height", default=int(full.height))
        full_w = int_attr(sbs, "width", default=int(full.width))
        eye_w = full_w // 2 if full_w > 0 else 1920
        vic = self._ensure_vic_bgr(eye_w=eye_w, eye_h=full_h)
        capture_seq = int_attr(sbs, "capture_seq", default=-1)
        try:
            bgr_prep = vic.prep_sbs_nv12(full, capture_seq=capture_seq)
            left_res, right_res, infer_ms = vic.infer_from_bgr(
                bgr_prep, parallel=parallel_eyes
            )
            preview_bgr = bgr_prep.left_bgr.copy() if with_gpu_preview else None
        finally:
            if release_sbs:
                sbs.release()
        return StereoInferResult(
            frame_id=self._frame_id,
            left=left_res,
            right=right_res,
            capture_ms=float(capture_ms),
            vic_prep_ms=float(bgr_prep.vic_ms),
            capture_seq=capture_seq,
            infer_parallel_ms=float(infer_ms),
            preview_bgr=preview_bgr,
            preview_ms=0.0,
        )

    def process_vic_bgr_prep(
        self,
        bgr_prep: Any,
        capture_ms: float,
        *,
        parallel_eyes: bool = True,
        prep_q_skipped: int = 0,
    ) -> StereoInferResult:
        """infer 路：TRT + post（prep 路已 letterbox 时跳过 LB）。"""
        from core.infer.vic_bgr_stereo import VicBgrStereoPrep

        if not isinstance(bgr_prep, VicBgrStereoPrep):
            raise TypeError(f"期望 VicBgrStereoPrep，收到 {type(bgr_prep)!r}")
        self._frame_id += 1
        # 勿用 bgr_prep.eye_w/h（竖装后是输出尺寸）去 recreate；
        # prep 路已按横幅源尺寸 _ensure_vic_bgr，此处复用同一实例。
        vic = self._vic_bgr
        if vic is None:
            raise RuntimeError("VicBgrStereoInfer 未初始化（prep 路应先创建）")
        left_res, right_res, infer_ms = vic.infer_from_bgr(
            bgr_prep, parallel=parallel_eyes
        )
        return StereoInferResult(
            frame_id=self._frame_id,
            left=left_res,
            right=right_res,
            capture_ms=float(capture_ms),
            vic_prep_ms=float(bgr_prep.vic_ms),
            capture_seq=int(bgr_prep.capture_seq),
            infer_parallel_ms=float(infer_ms),
            par_drain_ms=float(prep_q_skipped),
        )

    def _process_sbs_vic_bgr_overlap(
        self,
        sbs: Any,
        capture_ms: float,
        *,
        parallel_eyes: bool = True,
        release_sbs: bool = True,
        prefetch_sbs: Any | None = None,
    ) -> StereoInferResult | None:
        self._frame_id += 1
        full = sbs.full
        if full is None or full.dmabuf_fd is None:
            if release_sbs:
                sbs.release()
            return None

        full_h = int_attr(sbs, "height", default=int(full.height))
        full_w = int_attr(sbs, "width", default=int(full.width))
        eye_w = full_w // 2 if full_w > 0 else 1920
        vic = self._ensure_vic_bgr(eye_w=eye_w, eye_h=full_h)
        overlap = self._vic_bgr_overlap
        if overlap is None:
            overlap = _VicBgrOverlapState()
            self._vic_bgr_overlap = overlap

        from core.infer.vic_bgr_stereo import submit_vic_bgr_prep_async

        t_block = time.perf_counter()
        result_capture_seq = int_attr(sbs, "capture_seq", default=-1)
        vic_prep_ms = 0.0
        par_drain_ms = 0.0
        try:
            if overlap.inflight_future is not None:
                t_drain = time.perf_counter()
                fut = overlap.inflight_future
                overlap.inflight_future = None
                try:
                    bgr_prep = fut.result()
                except Exception:
                    if overlap.inflight_sbs is not None:
                        try:
                            overlap.inflight_sbs.release()
                        except Exception:
                            pass
                        overlap.inflight_sbs = None
                    raise
                if overlap.inflight_sbs is not None:
                    try:
                        overlap.inflight_sbs.release()
                    except Exception:
                        pass
                    overlap.inflight_sbs = None
                par_drain_ms = (time.perf_counter() - t_drain) * 1000.0
                result_capture_seq = int(bgr_prep.capture_seq)
                vic_prep_ms = float(bgr_prep.vic_ms)
                left_res, right_res, _ = vic.infer_from_bgr(
                    bgr_prep, parallel=parallel_eyes
                )
            else:
                capture_seq = int_attr(sbs, "capture_seq", default=-1)
                bgr_prep = vic.prep_sbs_nv12(full, capture_seq=capture_seq)
                result_capture_seq = capture_seq
                vic_prep_ms = float(bgr_prep.vic_ms)
                left_res, right_res, _ = vic.infer_from_bgr(
                    bgr_prep, parallel=parallel_eyes
                )

            if prefetch_sbs is not None:
                pf_full = getattr(prefetch_sbs, "full", None)
                if pf_full is not None and pf_full.dmabuf_fd is not None:
                    pf_seq = int_attr(prefetch_sbs, "capture_seq", default=-1)
                    overlap.inflight_sbs = prefetch_sbs
                    overlap.inflight_future = submit_vic_bgr_prep_async(
                        vic,
                        pf_full,
                        capture_seq=int(pf_seq),
                    )
        except Exception:
            self.reset_vic_bgr_overlap_state()
            raise
        finally:
            if release_sbs:
                sbs.release()

        infer_parallel_ms = (time.perf_counter() - t_block) * 1000.0
        return StereoInferResult(
            frame_id=self._frame_id,
            left=left_res,
            right=right_res,
            capture_ms=float(capture_ms),
            vic_prep_ms=vic_prep_ms,
            capture_seq=result_capture_seq,
            infer_parallel_ms=infer_parallel_ms,
            par_drain_ms=par_drain_ms,
        )

    def _prepared_to_eye_input(self, eye: Any) -> _PreparedEyeInput:
        return _PreparedEyeInput(
            ow=int_attr(eye, "ow", default=0),
            oh=int_attr(eye, "oh", default=0),
            geo=eye.geo,
            vic_unpad_ms=float(eye.vic_ms),
            egl_ms=float(eye.egl_ms),
            nv12_planes=eye.planes,
        )

    def _infer_prepared_eyes(
        self,
        prep: Any,
        *,
        parallel: bool = True,
    ) -> tuple[EyePoseResult, EyePoseResult, float]:
        prep_l = self._prepared_to_eye_input(prep.left)
        prep_r = self._prepared_to_eye_input(prep.right)
        t_block = time.perf_counter()

        if parallel:
            fut_feed_l = _infer_pool.submit(
                self._left_trt.feed_nv12_planes_to_input, prep.left.planes
            )
            fut_feed_r = _infer_pool.submit(
                self._right_trt.feed_nv12_planes_to_input, prep.right.planes
            )
            fut_feed_l.result()
            fut_feed_r.result()
        else:
            self._left_trt.feed_nv12_planes_to_input(prep.left.planes)
            self._right_trt.feed_nv12_planes_to_input(prep.right.planes)

        prep.release()

        prep_l.trt_prefed = True
        prep_r.trt_prefed = True

        if parallel:
            fut_l = _infer_pool.submit(self._run_trt, self._left_trt, prep_l)
            fut_r = _infer_pool.submit(self._run_trt, self._right_trt, prep_r)
            trt_l = fut_l.result()
            trt_r = fut_r.result()
            left_res = _infer_pool.submit(
                self._post_one_eye, prep_l, trt_l
            ).result()
            right_res = _infer_pool.submit(
                self._post_one_eye, prep_r, trt_r
            ).result()
        else:
            trt_l = self._run_trt(self._left_trt, prep_l)
            trt_r = self._run_trt(self._right_trt, prep_r)
            left_res = self._post_one_eye(prep_l, trt_l)
            right_res = self._post_one_eye(prep_r, trt_r)

        infer_parallel_ms = (time.perf_counter() - t_block) * 1000.0
        return left_res, right_res, infer_parallel_ms

    def process_prepared(
        self,
        prep: Any,
        *,
        parallel_eyes: bool = True,
        release: bool = True,
    ) -> StereoInferResult | None:
        """采集线程 VIC 预处理后：infer 仅 TRT + 后处理。"""
        if not self.trt_only:
            raise RuntimeError("process_prepared 需要 trt_only=True 的 pipeline")
        self._frame_id += 1
        preview_bgr = getattr(prep, "preview_bgr", None)
        preview_ms = float(getattr(prep, "preview_ms", 0.0))
        try:
            left_res, right_res, infer_ms = self._infer_prepared_eyes(
                prep,
                parallel=parallel_eyes,
            )
        except Exception:
            if release:
                prep.release()
            raise
        return StereoInferResult(
            frame_id=self._frame_id,
            left=left_res,
            right=right_res,
            capture_ms=float(prep.capture_ms),
            pull_ms=float(getattr(prep, "pull_ms", 0.0)),
            vic_prep_ms=float(getattr(prep, "vic_prep_ms", 0.0)),
            capture_seq=int_attr(prep, "capture_seq", default=-1),
            infer_parallel_ms=infer_ms,
            preview_bgr=preview_bgr,
            preview_ms=preview_ms,
        )

    def process_sbs(
        self,
        sbs: Any,
        capture_ms: float,
        *,
        parallel_eyes: bool = True,
        release_sbs: bool = True,
        with_gpu_preview: bool = False,
        pipeline_overlap: bool = False,
        prefetch_sbs: Any | None = None,
    ) -> StereoInferResult | None:
        full_w = int_attr(sbs, "width", default=0)
        full_h = int_attr(sbs, "height", default=0)
        eye_w = full_w // 2 if full_w > 0 else 1920

        if isinstance(sbs, Nv12SbsFrame) and sbs.full is not None and not sbs.eyes:
            full = sbs.full
            if full.dmabuf_fd is None:
                if release_sbs:
                    sbs.release()
                return None
            if (
                pipeline_overlap
                and self.gpu_letterbox
                and self._stereo_trt is not None
                and not with_gpu_preview
            ):
                return self._process_sbs_overlap(
                    sbs,
                    capture_ms,
                    parallel_eyes=parallel_eyes,
                    release_sbs=release_sbs,
                    prefetch_sbs=prefetch_sbs,
                )
            if (
                pipeline_overlap
                and self._use_vic_bgr
                and not with_gpu_preview
            ):
                return self._process_sbs_vic_bgr_overlap(
                    sbs,
                    capture_ms,
                    parallel_eyes=parallel_eyes,
                    release_sbs=release_sbs,
                    prefetch_sbs=prefetch_sbs,
                )

        self._frame_id += 1
        if isinstance(sbs, Nv12SbsFrame) and sbs.full is not None and not sbs.eyes:
            full = sbs.full
            if self._use_vic_bgr:
                return self._process_sbs_vic_bgr(
                    sbs,
                    capture_ms,
                    parallel_eyes=parallel_eyes,
                    release_sbs=release_sbs,
                    with_gpu_preview=with_gpu_preview,
                )
            if self.gpu_letterbox:
                left_eye = Nv12Frame(
                    y=full.y,
                    uv=full.uv,
                    width=eye_w,
                    height=full_h,
                    dmabuf_fd=full.dmabuf_fd,
                    pitch=full.pitch,
                    zero_copy=full.zero_copy,
                    _keepalive=full._keepalive,
                )
                right_eye = left_eye
                try:
                    left_res, right_res, infer_ms, preview_bgr, preview_ms = self._infer_eyes(
                        left_eye,
                        right_eye,
                        parallel=parallel_eyes,
                        full_w=full_w,
                        full_h=full_h,
                        eye_w=eye_w,
                        sbs_gpu_crop=True,
                        with_gpu_preview=with_gpu_preview,
                    )
                finally:
                    if release_sbs:
                        sbs.release()
                return StereoInferResult(
                    frame_id=self._frame_id,
                    left=left_res,
                    right=right_res,
                    capture_ms=capture_ms,
                    capture_seq=int_attr(sbs, "capture_seq", default=-1),
                    infer_parallel_ms=infer_ms,
                    preview_bgr=preview_bgr,
                    preview_ms=preview_ms,
                )
            if self._use_vic_letterbox:
                vic_prep = self._ensure_vic_sbs_prep(
                    full_w=full_w,
                    full_h=full_h,
                    eye_w=eye_w,
                )
                t_infer = time.perf_counter()
                prepared = vic_prep.prepare(
                    sbs,
                    with_preview=with_gpu_preview,
                    release_sbs=release_sbs,
                )
                lb_ms = float(prepared.vic_prep_ms)
                preview_bgr = prepared.preview_bgr
                preview_ms = float(prepared.preview_ms)
                try:
                    left_res, right_res, _trt_ms = self._infer_prepared_eyes(
                        prepared,
                        parallel=parallel_eyes,
                    )
                finally:
                    prepared.release()
                infer_ms = (time.perf_counter() - t_infer) * 1000.0
                return StereoInferResult(
                    frame_id=self._frame_id,
                    left=left_res,
                    right=right_res,
                    capture_ms=float(capture_ms),
                    vic_prep_ms=lb_ms,
                    capture_seq=int_attr(sbs, "capture_seq", default=-1),
                    infer_parallel_ms=infer_ms,
                    preview_bgr=preview_bgr,
                    preview_ms=preview_ms,
                )
            sbs.release()
            raise RuntimeError(
                "单路 SBS 采集需 infer_prep=gpu|vic_bgr（默认 vic_bgr）"
            )

        left_eye = sbs.eyes.get("left")
        right_eye = sbs.eyes.get("right")
        if left_eye is None or right_eye is None:
            sbs.release()
            return None
        try:
            left_res, right_res, infer_ms, _, _ = self._infer_eyes(
                left_eye, right_eye, parallel=parallel_eyes
            )
        finally:
            if release_sbs:
                sbs.release()
            else:
                sbs.release_eyes()

        return StereoInferResult(
            frame_id=self._frame_id,
            left=left_res,
            right=right_res,
            capture_ms=capture_ms,
            capture_seq=int_attr(sbs, "capture_seq", default=-1),
            infer_parallel_ms=infer_ms,
        )

    def process_capture(
        self,
        cap: JetsonGstNvmmCapture,
        *,
        parallel_eyes: bool = True,
    ) -> StereoInferResult | None:
        t0 = time.perf_counter()
        ok, sbs = cap.read_nv12()
        capture_ms = (time.perf_counter() - t0) * 1000.0
        if not ok or sbs is None:
            return None
        return self.process_sbs(
            sbs, capture_ms, parallel_eyes=parallel_eyes, release_sbs=True
        )

    def close(self) -> None:
        self.reset_overlap_state()
        self._overlap = None
        self._vic_bgr_overlap = None
        if self._vic_sbs_prep is not None:
            self._vic_sbs_prep.release()
            self._vic_sbs_prep = None
        if self._vic_bgr is not None:
            self._vic_bgr.close()
            self._vic_bgr = None
        if self._left_vic is not None:
            self._left_vic.release()
        if self._right_vic is not None:
            self._right_vic.release()
        try:
            from platform.jetson.nvbuf_egl_cuda import release_shared_egl_cuda_mapper

            release_shared_egl_cuda_mapper()
        except Exception:
            pass
