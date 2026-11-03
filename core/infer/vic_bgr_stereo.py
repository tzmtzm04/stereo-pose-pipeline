"""实时 infer：prep 路 VIC→BGR→letterbox；infer 路 TRT+post（overlap 时分线程）。"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from core.infer.postprocess import postprocess_v26
from core.infer.stereo_pipeline import EyePoseResult, _TrtEyeOutput
from platform.jetson.letterbox_util import LetterboxGeometry, letterbox_bgr_cpu
from platform.jetson.nv12_frame import Nv12Frame
from platform.jetson.trt import YOLO26PoseTRTRunner

_infer_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="route2_vic_bgr_trt")
_vic_prep_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="route2_vic_bgr_prep")


@dataclass
class VicBgrStereoPrep:
    """prep 路输出：全幅 BGR 或已 letterbox 的双眼 canvas。"""

    capture_seq: int
    vic_ms: float
    eye_w: int
    eye_h: int
    lb_ms: float = 0.0
    left_bgr: np.ndarray | None = None
    right_bgr: np.ndarray | None = None
    left_eye: _BgrEyePrep | None = None
    right_eye: _BgrEyePrep | None = None

    @property
    def letterboxed(self) -> bool:
        return self.left_eye is not None and self.right_eye is not None


@dataclass(frozen=True)
class VicBgrInferPrepItem:
    """prep 路 → infer 路（letterbox 已在 prep 或全幅 BGR，不含 SBS fd）。"""

    prep: VicBgrStereoPrep
    capture_ms: float
    capture_skipped: int = 0


@dataclass
class _BgrEyePrep:
    ow: int
    oh: int
    geo: LetterboxGeometry
    canvas: np.ndarray
    lb_ms: float = 0.0
    vic_ms: float = 0.0


class VicBgrStereoInfer:
    """prep：SBS fd → VIC 裁切[+旋转] BGR（+ 可选 letterbox）；infer：TRT + post。"""

    def __init__(
        self,
        engine_path: Path | str,
        *,
        conf_threshold: float = 0.25,
        eye_w: int = 1920,
        eye_h: int = 1080,
        input_size: int = 640,
        eye_rotate: str = "none",
    ) -> None:
        from core.geometry.eye_mount_rotate import parse_eye_mount_rotate, rotated_eye_hw
        from core.offline.sbs_nv12_vic_split import SbsNv12VicSplitter

        self._conf = float(conf_threshold)
        self._input_size = int(input_size)
        self._src_eye_w = int(eye_w)
        self._src_eye_h = int(eye_h)
        self._eye_rotate = parse_eye_mount_rotate(eye_rotate)
        out_h, out_w = rotated_eye_hw(
            self._src_eye_h, self._src_eye_w, self._eye_rotate
        )
        self._eye_w = int(out_w)
        self._eye_h = int(out_h)
        self._splitter = SbsNv12VicSplitter(
            self._src_eye_w, self._src_eye_h, eye_rotate=self._eye_rotate
        )
        self._left_trt = YOLO26PoseTRTRunner(engine_path)
        self._right_trt = YOLO26PoseTRTRunner(engine_path)
        if (
            self._left_trt.input_w != self._input_size
            or self._left_trt.input_h != self._input_size
        ):
            raise ValueError(
                f"engine 输入须 {self._input_size}x{self._input_size}，"
                f"收到 {self._left_trt.input_w}x{self._left_trt.input_h}"
            )

    def close(self) -> None:
        self._splitter.close()

    def prep_sbs_nv12(
        self,
        full: Nv12Frame,
        *,
        capture_seq: int = -1,
    ) -> VicBgrStereoPrep:
        """prep 路：VIC 裁半[+Rotate90] → 双眼 BGR（竖装时已是竖图）。"""
        if full.dmabuf_fd is None:
            raise ValueError("Nv12Frame 需要 dmabuf_fd")
        t_vic = time.perf_counter()
        left_bgr, right_bgr = self._splitter.split_eye_bgr(full)
        left_bgr = np.ascontiguousarray(left_bgr)
        right_bgr = np.ascontiguousarray(right_bgr)
        vic_ms = (time.perf_counter() - t_vic) * 1000.0
        return VicBgrStereoPrep(
            capture_seq=int(capture_seq),
            left_bgr=left_bgr,
            right_bgr=right_bgr,
            vic_ms=float(vic_ms),
            eye_w=self._eye_w,
            eye_h=self._eye_h,
        )

    def prep_sbs_nv12_letterbox(
        self,
        full: Nv12Frame,
        *,
        capture_seq: int = -1,
    ) -> VicBgrStereoPrep:
        """prep 路：VIC→BGR→CPU letterbox（供 prep 线程；infer 仅 TRT+post）。"""
        bgr_prep = self.prep_sbs_nv12(full, capture_seq=capture_seq)
        half_vic = float(bgr_prep.vic_ms) * 0.5
        assert bgr_prep.left_bgr is not None and bgr_prep.right_bgr is not None
        eye_prep_l = self._letterbox_eye(bgr_prep.left_bgr, vic_ms=half_vic)
        eye_prep_r = self._letterbox_eye(bgr_prep.right_bgr, vic_ms=half_vic)
        lb_ms = float(eye_prep_l.lb_ms + eye_prep_r.lb_ms)
        return VicBgrStereoPrep(
            capture_seq=int(bgr_prep.capture_seq),
            vic_ms=float(bgr_prep.vic_ms),
            lb_ms=lb_ms,
            eye_w=int(bgr_prep.eye_w),
            eye_h=int(bgr_prep.eye_h),
            left_eye=eye_prep_l,
            right_eye=eye_prep_r,
        )

    def _letterbox_eye(self, bgr: np.ndarray, *, vic_ms: float) -> _BgrEyePrep:
        t0 = time.perf_counter()
        canvas, geo = letterbox_bgr_cpu(bgr, self._input_size, self._input_size)
        lb_ms = (time.perf_counter() - t0) * 1000.0
        return _BgrEyePrep(
            ow=int(bgr.shape[1]),
            oh=int(bgr.shape[0]),
            geo=geo,
            canvas=canvas,
            lb_ms=lb_ms,
            vic_ms=float(vic_ms),
        )

    def _run_trt(
        self,
        runner: YOLO26PoseTRTRunner,
        prep: _BgrEyePrep,
    ) -> _TrtEyeOutput:
        raw = runner.infer_bgr_letterboxed(prep.canvas)
        return _TrtEyeOutput(
            raw=raw,
            pre_ms=prep.lb_ms,
            infer_ms=float(runner.last_infer_ms),
        )

    def _post_one(
        self,
        prep: _BgrEyePrep,
        trt_out: _TrtEyeOutput,
    ) -> EyePoseResult:
        t_post = time.perf_counter()
        poses = postprocess_v26(
            trt_out.raw,
            (prep.ow, prep.oh),
            letterbox_geometry=prep.geo,
            conf_threshold=self._conf,
        )
        post_ms = (time.perf_counter() - t_post) * 1000.0
        return EyePoseResult(
            poses=poses,
            vic_unpad_ms=prep.vic_ms,
            pre_ms=trt_out.pre_ms,
            infer_ms=trt_out.infer_ms,
            post_ms=post_ms,
        )

    def infer_from_bgr(
        self,
        prep: VicBgrStereoPrep,
        *,
        parallel: bool = True,
    ) -> tuple[EyePoseResult, EyePoseResult, float]:
        """infer 路：TRT + post；未 letterbox 的 prep 在本线程补 LB。"""
        t_block = time.perf_counter()
        if prep.letterboxed:
            eye_prep_l = prep.left_eye
            eye_prep_r = prep.right_eye
            assert eye_prep_l is not None and eye_prep_r is not None
        else:
            half_vic = float(prep.vic_ms) * 0.5
            assert prep.left_bgr is not None and prep.right_bgr is not None
            eye_prep_l = self._letterbox_eye(prep.left_bgr, vic_ms=half_vic)
            eye_prep_r = self._letterbox_eye(prep.right_bgr, vic_ms=half_vic)

        if parallel:
            fut_l = _infer_pool.submit(self._run_trt, self._left_trt, eye_prep_l)
            fut_r = _infer_pool.submit(self._run_trt, self._right_trt, eye_prep_r)
            trt_l = fut_l.result()
            trt_r = fut_r.result()
            left_res = self._post_one(eye_prep_l, trt_l)
            right_res = self._post_one(eye_prep_r, trt_r)
        else:
            trt_l = self._run_trt(self._left_trt, eye_prep_l)
            trt_r = self._run_trt(self._right_trt, eye_prep_r)
            left_res = self._post_one(eye_prep_l, trt_l)
            right_res = self._post_one(eye_prep_r, trt_r)

        infer_ms = (time.perf_counter() - t_block) * 1000.0
        return left_res, right_res, infer_ms

    def infer_sbs_nv12(
        self,
        full: Nv12Frame,
        *,
        parallel: bool = True,
        with_preview: bool = False,
        capture_seq: int = -1,
    ) -> tuple[EyePoseResult, EyePoseResult, float, Optional[np.ndarray]]:
        """同步全路径（无 overlap）。"""
        bgr_prep = self.prep_sbs_nv12(full, capture_seq=capture_seq)
        left_res, right_res, infer_ms = self.infer_from_bgr(bgr_prep, parallel=parallel)
        preview = (
            bgr_prep.left_bgr.copy()
            if with_preview and bgr_prep.left_bgr is not None
            else None
        )
        return left_res, right_res, float(bgr_prep.vic_ms), preview


def submit_vic_bgr_prep_async(
    vic: VicBgrStereoInfer,
    full: Nv12Frame,
    *,
    capture_seq: int,
):
    """overlap：在专用单线程池上 prep 下一帧（VIC 非线程安全，仅 1 inflight）。"""
    return _vic_prep_pool.submit(
        vic.prep_sbs_nv12,
        full,
        capture_seq=int(capture_seq),
    )
