"""YOLO26-detect TensorRT runner（ctypes CUDA / Jetson DLA，无 PyTorch）。"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import numpy as np

from app.paths import DEFAULT_YOLO26_DET_ENGINE
from core.infer.yolo26_det_postprocess import best_person_bbox, postprocess_v26_detect
from platform.jetson.letterbox_util import LetterboxGeometry, letterbox_bgr_cpu
from platform.jetson.trt.yolo26_runner import YOLO26PoseTRTRunner


class YOLO26DetTRTRunner(YOLO26PoseTRTRunner):
    """输入 NCHW float32 RGB 640×640；输出 V26 detect [1,300,6]。"""

    def __init__(
        self,
        engine_path: Union[str, Path] = DEFAULT_YOLO26_DET_ENGINE,
        *,
        dla_core: int | None = None,
    ) -> None:
        super().__init__(engine_path, dla_core=dla_core)
        dev = f"DLA{self.dla_core}" if self.dla_core is not None else "GPU"
        print(
            f"YOLO26DetTRT: {self.engine_path.name} {dev} "
            f"{self.input_w}x{self.input_h}",
            flush=True,
        )

    def detect_person_bgr(
        self,
        bgr: np.ndarray,
        *,
        conf_threshold: float = 0.25,
    ) -> tuple[tuple[int, int, int, int], float] | None:
        canvas, geo = letterbox_bgr_cpu(bgr, self.input_w, self.input_h)
        raw = self.infer_bgr_letterboxed(canvas)
        return self.detect_person_from_raw(
            raw,
            (bgr.shape[1], bgr.shape[0]),
            geo,
            conf_threshold=conf_threshold,
        )

    def detect_person_from_raw(
        self,
        raw: np.ndarray,
        orig_wh: tuple[int, int],
        geo: LetterboxGeometry,
        *,
        conf_threshold: float = 0.25,
    ) -> tuple[tuple[int, int, int, int], float] | None:
        dets = postprocess_v26_detect(
            raw,
            orig_wh,
            letterbox_geometry=geo,
            conf_threshold=conf_threshold,
        )
        return best_person_bbox(dets)
