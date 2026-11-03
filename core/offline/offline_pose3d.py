"""离线 pose infer：VIC 串行裁切+letterbox（默认）或 GPU letterbox / CPU BGR 回退。"""

from __future__ import annotations

import queue
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from core.infer.postprocess import postprocess_v26
from platform.jetson.letterbox_util import LetterboxGeometry
from platform.jetson.nv12_frame import Nv12Frame
from platform.jetson.trt.yolo26_runner import YOLO26PoseTRTRunner
from platform.jetson.trt.yolo26_stereo_runner import YOLO26PoseStereoTRTRunner

_infer_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="offline_trt")


def offline_vic_available() -> bool:
    try:
        from platform.jetson.cuda.nv12_rgb import cuda_nv12_rgb_available
        from platform.jetson.vic import route2_nvbuf_vic_available

        return bool(route2_nvbuf_vic_available() and cuda_nv12_rgb_available())
    except Exception:
        return False


def offline_gpu_letterbox_available() -> bool:
    try:
        from platform.jetson.route2_gpu_preprocess import gpu_letterbox_available

        return bool(gpu_letterbox_available())
    except Exception:
        return False


def offline_hw_available() -> bool:
    """NVDEC + VIC 或 GPU letterbox 任一可用。"""
    return offline_vic_available() or offline_gpu_letterbox_available()


def offline_gpu_available() -> bool:
    """兼容旧名：默认 VIC 路径可用即 True。"""
    return offline_hw_available()


def _best_pose(poses: list[dict]) -> dict | None:
    if not poses:
        return None
    return max(poses, key=lambda p: float(p.get("conf", 0.0)))


@dataclass(frozen=True)
class EyeInferResult:
    det: dict | None
    infer_ms: float
    prep_ms: float = 0.0


@dataclass(frozen=True)
class _EyePrep:
    geo: LetterboxGeometry
    ow: int
    oh: int
    prep_ms: float
    nv12_planes: Any = None


@dataclass
class StereoTrtPrep:
    """VIC 已完成 letterbox，TRT 线程仅 feed + infer。"""

    capture_seq: int
    prep_l: _EyePrep
    prep_r: _EyePrep
    prep_ms: float
    decode_ms: float
    _vic_slot: tuple[Any, Any] | None = None
    _pool: Any | None = None

    def release_vic(self) -> None:
        if self._pool is not None and self._vic_slot is not None:
            self._pool.release(self._vic_slot)
            self._vic_slot = None
            self._pool = None


class _VicSlotPool:
    """VIC 输出缓冲池：prep_q 在途时避免覆盖 letterbox 平面。"""

    def __init__(
        self,
        pool_size: int,
        *,
        input_size: int,
        eye_w: int,
        eye_h: int,
    ) -> None:
        from platform.jetson.vic import Route2VicPreprocessor

        n = max(2, int(pool_size))
        self._free: queue.Queue = queue.Queue()
        self._slots: list[tuple[Any, Any]] = []
        for _ in range(n):
            left = Route2VicPreprocessor((input_size, input_size))
            right = Route2VicPreprocessor((input_size, input_size))
            left.warm_up(eye_w, eye_h)
            right.warm_up(eye_w, eye_h)
            slot = (left, right)
            self._slots.append(slot)
            self._free.put(slot)

    def acquire(self) -> tuple[Any, Any]:
        return self._free.get()

    def release(self, slot: tuple[Any, Any]) -> None:
        self._free.put(slot)

    def close(self) -> None:
        for left, right in self._slots:
            left.release()
            right.release()
        self._slots.clear()


@dataclass
class StereoBgrPrep:
    """NVDEC + VIC 裁切后的 BGR（--cpu infer）。"""

    capture_seq: int
    left_bgr: np.ndarray
    right_bgr: np.ndarray
    decode_ms: float
    prep_ms: float = 0.0



class OfflineBgrPoseInfer:
    """CPU BGR letterbox + 双眼串行 TRT（--cpu 回退）。"""

    def __init__(
        self,
        engine_path: Path,
        *,
        conf_threshold: float = 0.25,
        input_size: int = 640,
    ) -> None:
        from platform.jetson.letterbox_util import letterbox_bgr_cpu

        self._letterbox_bgr_cpu = letterbox_bgr_cpu
        self._conf = float(conf_threshold)
        self._input_size = int(input_size)
        self._left = YOLO26PoseTRTRunner(engine_path)
        self._right = YOLO26PoseTRTRunner(engine_path)
        if (
            self._left.input_w != self._input_size
            or self._left.input_h != self._input_size
        ):
            raise ValueError(
                f"engine 输入须 {self._input_size}x{self._input_size}，"
                f"收到 {self._left.input_w}x{self._left.input_h}"
            )

    def _infer_one_eye(
        self,
        runner: YOLO26PoseTRTRunner,
        bgr: np.ndarray,
    ) -> EyeInferResult:
        canvas, geo = self._letterbox_bgr_cpu(
            bgr, self._input_size, self._input_size
        )
        raw = runner.infer_bgr_letterboxed(canvas)
        det = _det_from_raw(raw, (bgr.shape[1], bgr.shape[0]), geo, self._conf)
        return EyeInferResult(det=det, infer_ms=runner.last_infer_ms)

    def infer_stereo(
        self,
        left_bgr: np.ndarray,
        right_bgr: np.ndarray,
    ) -> tuple[Optional[dict], Optional[dict], float]:
        left = self._infer_one_eye(self._left, left_bgr)
        right = self._infer_one_eye(self._right, right_bgr)
        infer_ms = float(left.infer_ms + right.infer_ms)
        return left.det, right.det, infer_ms

    def close(self) -> None:
        pass


class OfflineVicPoseInfer:
    """decode 线程：VIC letterbox；TRT 线程：feed + infer。"""

    def __init__(
        self,
        engine_path: Path,
        *,
        conf_threshold: float = 0.25,
        input_size: int = 640,
        parallel_eyes: bool = True,
        eye_w: int = 1920,
        eye_h: int = 1080,
        prep_pool_size: int = 3,
    ) -> None:
        from platform.jetson.vic import route2_nvbuf_vic_available

        if not route2_nvbuf_vic_available():
            raise RuntimeError("NvBuf VIC 不可用")
        self._conf = float(conf_threshold)
        self._input_size = int(input_size)
        self._parallel = bool(parallel_eyes)
        self._eye_w = int(eye_w)
        self._eye_h = int(eye_h)
        self._pool = _VicSlotPool(
            prep_pool_size,
            input_size=self._input_size,
            eye_w=self._eye_w,
            eye_h=self._eye_h,
        )
        self._stereo_mode = False
        self._stereo_trt: YOLO26PoseStereoTRTRunner | None = None
        self._left_trt: YOLO26PoseTRTRunner | None = None
        self._right_trt: YOLO26PoseTRTRunner | None = None
        try:
            self._stereo_trt = YOLO26PoseStereoTRTRunner(engine_path)
            self._stereo_mode = True
            probe = self._stereo_trt
        except RuntimeError:
            self._left_trt = YOLO26PoseTRTRunner(engine_path)
            self._right_trt = YOLO26PoseTRTRunner(engine_path)
            probe = self._left_trt
        if probe.input_w != self._input_size or probe.input_h != self._input_size:
            raise ValueError(
                f"engine 输入须 {self._input_size}x{self._input_size}，"
                f"收到 {probe.input_w}x{probe.input_h}"
            )

    def _prep_eye_roi(
        self,
        vic: Any,
        sbs_fd: int,
        *,
        x1: int,
        x2: int,
        full_h: int,
    ) -> _EyePrep:
        eye_w = self._eye_w
        eye_h = self._eye_h
        planes, prof = vic.prepare_nv12_cuda_from_roi_fd(
            int(sbs_fd),
            int(x1),
            0,
            int(x2),
            int(full_h),
            eye_w,
            eye_h,
        )
        geo = vic.last_geometry
        if geo is None:
            raise RuntimeError("VIC letterbox geometry 缺失")
        return _EyePrep(
            geo=geo,
            ow=eye_w,
            oh=eye_h,
            prep_ms=prof.vic_total_ms,
            nv12_planes=planes,
        )

    def _run_trt_one(
        self,
        trt: YOLO26PoseTRTRunner,
        prep: _EyePrep,
    ) -> tuple[np.ndarray, float]:
        assert prep.nv12_planes is not None
        trt.feed_nv12_planes_to_input(prep.nv12_planes)
        raw = trt.infer_trt_only()
        return raw, float(trt.last_infer_ms)

    def _feed_stereo_batch(self, prep: StereoTrtPrep) -> None:
        assert self._stereo_trt is not None
        assert prep.prep_l.nv12_planes is not None
        assert prep.prep_r.nv12_planes is not None
        st = self._stereo_trt

        def _feed(batch_idx: int, eye_prep: _EyePrep) -> None:
            assert eye_prep.nv12_planes is not None
            st.feed_nv12_planes_to_input_at(batch_idx, eye_prep.nv12_planes)

        if self._parallel:
            fut_l = _infer_pool.submit(_feed, 0, prep.prep_l)
            fut_r = _infer_pool.submit(_feed, 1, prep.prep_r)
            fut_l.result()
            fut_r.result()
        else:
            _feed(0, prep.prep_l)
            _feed(1, prep.prep_r)

    def _infer_stereo_batch(
        self,
        prep: StereoTrtPrep,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        assert self._stereo_trt is not None
        t0 = time.perf_counter()
        self._feed_stereo_batch(prep)
        raw_b = self._stereo_trt.infer_trt_only()
        trt_ms = (time.perf_counter() - t0) * 1000.0
        return raw_b[0], raw_b[1], trt_ms

    def prep_sbs_nv12(
        self,
        sbs: Nv12Frame,
        capture_seq: int,
        decode_ms: float,
    ) -> StereoTrtPrep:
        """NVDEC 输出 fd → VIC 双眼 letterbox（decode 线程调用）。"""
        if sbs.dmabuf_fd is None:
            raise ValueError("Nv12Frame 需要 dmabuf_fd")
        left_vic, right_vic = self._pool.acquire()
        t0 = time.perf_counter()
        try:
            full_w, full_h = int(sbs.width), int(sbs.height)
            eye_w = full_w // 2
            fd = int(sbs.dmabuf_fd)
            prep_l = self._prep_eye_roi(
                left_vic, fd, x1=0, x2=eye_w, full_h=full_h
            )
            prep_r = self._prep_eye_roi(
                right_vic, fd, x1=eye_w, x2=full_w, full_h=full_h
            )
            prep_ms = (time.perf_counter() - t0) * 1000.0
            return StereoTrtPrep(
                capture_seq=int(capture_seq),
                prep_l=prep_l,
                prep_r=prep_r,
                prep_ms=prep_ms,
                decode_ms=float(decode_ms),
                _vic_slot=(left_vic, right_vic),
                _pool=self._pool,
            )
        except Exception:
            self._pool.release((left_vic, right_vic))
            raise

    def infer_prepared(
        self,
        prep: StereoTrtPrep,
    ) -> tuple[Optional[dict], Optional[dict], float]:
        """TRT 线程：NV12→RGB + infer + postprocess。"""
        t0 = time.perf_counter()
        if self._stereo_mode:
            raw_l, raw_r, _ = self._infer_stereo_batch(prep)
        elif self._parallel:
            assert self._left_trt is not None and self._right_trt is not None
            fut_l = _infer_pool.submit(self._run_trt_one, self._left_trt, prep.prep_l)
            fut_r = _infer_pool.submit(self._run_trt_one, self._right_trt, prep.prep_r)
            raw_l, _ = fut_l.result()
            raw_r, _ = fut_r.result()
        else:
            assert self._left_trt is not None and self._right_trt is not None
            raw_l, _ = self._run_trt_one(self._left_trt, prep.prep_l)
            raw_r, _ = self._run_trt_one(self._right_trt, prep.prep_r)
        left = _det_from_raw(
            raw_l, (prep.prep_l.ow, prep.prep_l.oh), prep.prep_l.geo, self._conf
        )
        right = _det_from_raw(
            raw_r, (prep.prep_r.ow, prep.prep_r.oh), prep.prep_r.geo, self._conf
        )
        trt_ms = (time.perf_counter() - t0) * 1000.0
        return left, right, trt_ms

    def infer_stereo_sbs(
        self,
        sbs: Nv12Frame,
        *,
        capture_seq: int = -1,
        decode_ms: float = 0.0,
    ) -> tuple[Optional[dict], Optional[dict], float]:
        """串行模式（--no-pipeline）：prep + TRT 同线程。"""
        prep = self.prep_sbs_nv12(sbs, capture_seq, decode_ms)
        try:
            return self.infer_prepared(prep)
        finally:
            prep.release_vic()

    def close(self) -> None:
        self._pool.close()


class OfflineGpuPoseInfer:
    """SBS fd → EGL + CUDA letterbox 直写 TRT batch（对齐实时 stereo_pipeline）。"""

    def __init__(
        self,
        engine_path: Path,
        *,
        conf_threshold: float = 0.25,
        input_size: int = 640,
        parallel_eyes: bool = True,
        stereo_letterbox_fused: bool = True,
    ) -> None:
        from platform.jetson.cuda.nv12_rgb import gpu_stereo_sbs_letterbox_cuda_available
        from platform.jetson.route2_gpu_preprocess import (
            Route2GpuPreprocessor,
            gpu_letterbox_available,
        )

        if not gpu_letterbox_available():
            raise RuntimeError(
                "GPU letterbox 需要 route2_nv12_rgb.so: "
                "bash scripts/route2/build_cuda_nv12_rgb.sh"
            )
        self._conf = float(conf_threshold)
        self._input_size = int(input_size)
        self._parallel = bool(parallel_eyes)
        self._last_prep_ms = 0.0
        self._last_trt_ms = 0.0
        self._stereo_trt: YOLO26PoseStereoTRTRunner | None = None
        self._left_trt: YOLO26PoseTRTRunner | None = None
        self._right_trt: YOLO26PoseTRTRunner | None = None
        try:
            self._stereo_trt = YOLO26PoseStereoTRTRunner(engine_path)
            self._stereo_mode = True
            probe = self._stereo_trt
        except RuntimeError:
            self._left_trt = YOLO26PoseTRTRunner(engine_path)
            self._right_trt = YOLO26PoseTRTRunner(engine_path)
            self._stereo_mode = False
            probe = self._left_trt
        self._left_gpu = Route2GpuPreprocessor((self._input_size, self._input_size))
        self._right_gpu = Route2GpuPreprocessor((self._input_size, self._input_size))
        self._stereo_fused = (
            bool(stereo_letterbox_fused)
            and self._stereo_mode
            and gpu_stereo_sbs_letterbox_cuda_available()
        )
        if probe.input_w != self._input_size or probe.input_h != self._input_size:
            raise ValueError(
                f"engine 输入须 {self._input_size}x{self._input_size}，"
                f"收到 {probe.input_w}x{probe.input_h}"
            )
        fused_tag = "fused" if self._stereo_fused else "dual_crop"
        batch_tag = f"batch={probe.batch_size} " if probe.batch_size > 1 else ""
        print(
            f"OfflineGpuPoseInfer: {batch_tag}letterbox→trt_input ({fused_tag})",
            flush=True,
        )

    @property
    def last_prep_ms(self) -> float:
        return float(self._last_prep_ms)

    @property
    def last_trt_ms(self) -> float:
        return float(self._last_trt_ms)

    def _prep_stereo(
        self,
        sbs: Nv12Frame,
        *,
        full_w: int,
        full_h: int,
        eye_w: int,
        eye_h: int,
    ) -> tuple[_EyePrep, _EyePrep]:
        from platform.jetson.sbs_gpu_egl_session import SbsGpuEglSession

        if sbs.dmabuf_fd is None:
            raise ValueError("Nv12Frame 需要 dmabuf_fd")
        st = self._stereo_trt
        lb_kw = dict(crop_y=0, crop_h=eye_h)
        lb_stream = st.stream.handle if st is not None else self._left_trt.stream.handle
        egl_ms = 0.0
        with SbsGpuEglSession.open(
            int(sbs.dmabuf_fd),
            full_w,
            full_h,
            stream_handle=lb_stream,
        ) as session:
            egl_ms = session.egl_bind_ms
            if self._stereo_fused and st is not None:
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
                st.stream.sync()
            elif st is not None:
                stream_l = st.stream.handle
                stream_r = st._letterbox_aux_stream.handle
                if self._parallel:
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
                else:
                    geo_l, lb_l = session.letterbox_crop_to_trt(
                        st,
                        self._left_gpu,
                        crop_x=0,
                        crop_w=eye_w,
                        stream_handle=stream_l,
                        dst_ptr=st.input_device_ptr_at(0),
                        **lb_kw,
                    )
                    geo_r, lb_r = session.letterbox_crop_to_trt(
                        st,
                        self._right_gpu,
                        crop_x=eye_w,
                        crop_w=eye_w,
                        stream_handle=stream_r,
                        dst_ptr=st.input_device_ptr_at(1),
                        **lb_kw,
                    )
                st.stream.sync()
                st._letterbox_aux_stream.sync()
            else:
                assert self._left_trt is not None and self._right_trt is not None
                stream_l = self._left_trt.stream.handle
                stream_r = self._right_trt.stream.handle
                if self._parallel:
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
                else:
                    geo_l, lb_l = session.letterbox_crop_to_trt(
                        self._left_trt,
                        self._left_gpu,
                        crop_x=0,
                        crop_w=eye_w,
                        stream_handle=stream_l,
                        **lb_kw,
                    )
                    geo_r, lb_r = session.letterbox_crop_to_trt(
                        self._right_trt,
                        self._right_gpu,
                        crop_x=eye_w,
                        crop_w=eye_w,
                        stream_handle=stream_r,
                        **lb_kw,
                    )
                self._left_trt.stream.sync()
                self._right_trt.stream.sync()
        prep_l = _EyePrep(
            geo=geo_l,
            ow=eye_w,
            oh=eye_h,
            prep_ms=lb_l + egl_ms * 0.5,
        )
        prep_r = _EyePrep(
            geo=geo_r,
            ow=eye_w,
            oh=eye_h,
            prep_ms=lb_r + egl_ms * 0.5,
        )
        return prep_l, prep_r

    def _infer_from_prep(
        self,
        prep_l: _EyePrep,
        prep_r: _EyePrep,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        t0 = time.perf_counter()
        if self._stereo_mode:
            assert self._stereo_trt is not None
            raw_b = self._stereo_trt.infer_trt_only()
            infer_ms = (time.perf_counter() - t0) * 1000.0
            return raw_b[0], raw_b[1], infer_ms
        assert self._left_trt is not None and self._right_trt is not None
        if self._parallel:
            fut_l = _infer_pool.submit(self._left_trt.infer_trt_only)
            fut_r = _infer_pool.submit(self._right_trt.infer_trt_only)
            raw_l = fut_l.result()
            raw_r = fut_r.result()
        else:
            raw_l = self._left_trt.infer_trt_only()
            raw_r = self._right_trt.infer_trt_only()
        infer_ms = (time.perf_counter() - t0) * 1000.0
        return raw_l, raw_r, infer_ms

    def infer_stereo_sbs(
        self,
        sbs: Nv12Frame,
        *,
        capture_seq: int = -1,
        decode_ms: float = 0.0,
    ) -> tuple[Optional[dict], Optional[dict], float]:
        _ = capture_seq, decode_ms
        full_w, full_h = int(sbs.width), int(sbs.height)
        eye_w = full_w // 2
        eye_h = full_h
        prep_l, prep_r = self._prep_stereo(
            sbs,
            full_w=full_w,
            full_h=full_h,
            eye_w=eye_w,
            eye_h=eye_h,
        )
        self._last_prep_ms = float(prep_l.prep_ms + prep_r.prep_ms)
        t_infer = time.perf_counter()
        raw_l, raw_r, infer_ms = self._infer_from_prep(prep_l, prep_r)
        left = _det_from_raw(raw_l, (prep_l.ow, prep_l.oh), prep_l.geo, self._conf)
        right = _det_from_raw(raw_r, (prep_r.ow, prep_r.oh), prep_r.geo, self._conf)
        post_ms = (time.perf_counter() - t_infer) * 1000.0 - infer_ms
        self._last_trt_ms = float(infer_ms + max(0.0, post_ms))
        block_ms = self._last_prep_ms + self._last_trt_ms
        return left, right, block_ms

    def close(self) -> None:
        pass


def _det_from_raw(
    raw: np.ndarray,
    orig_wh: tuple[int, int],
    geo: LetterboxGeometry,
    conf: float,
    *,
    orig_map_mode: str = "float_pad",
) -> dict | None:
    poses = postprocess_v26(
        raw,
        orig_wh,
        letterbox_geometry=geo,
        conf_threshold=conf,
        orig_map_mode=str(orig_map_mode),
    )
    det = _best_pose(poses)
    if det is not None:
        kp = np.asarray(det["keypoints"], dtype=np.float32).copy()
        det = dict(det)
        det["keypoints"] = kp
    return det


def create_offline_pose_infer(
    engine_path: Path,
    *,
    use_gpu: bool = True,
    gpu_letterbox: bool | None = None,
    conf_threshold: float = 0.25,
    input_size: int = 640,
    parallel_eyes: bool = True,
    eye_w: int = 1920,
    eye_h: int = 1080,
    prep_pool_size: int = 3,
) -> OfflineVicPoseInfer | OfflineGpuPoseInfer | OfflineBgrPoseInfer:
    if not use_gpu:
        return OfflineBgrPoseInfer(
            engine_path,
            conf_threshold=conf_threshold,
            input_size=input_size,
        )
    prefer_gpu = True if gpu_letterbox is None else bool(gpu_letterbox)
    if prefer_gpu and offline_gpu_letterbox_available():
        return OfflineGpuPoseInfer(
            engine_path,
            conf_threshold=conf_threshold,
            input_size=input_size,
            parallel_eyes=parallel_eyes,
        )
    if offline_vic_available():
        return OfflineVicPoseInfer(
            engine_path,
            conf_threshold=conf_threshold,
            input_size=input_size,
            parallel_eyes=parallel_eyes,
            eye_w=eye_w,
            eye_h=eye_h,
            prep_pool_size=max(3, int(prep_pool_size)),
        )
    return OfflineBgrPoseInfer(
        engine_path,
        conf_threshold=conf_threshold,
        input_size=input_size,
    )
