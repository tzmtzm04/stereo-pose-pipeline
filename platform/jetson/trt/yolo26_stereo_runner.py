"""YOLO26n-pose batch=2 TensorRT（双眼一次 enqueue）。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import numpy as np

from platform.jetson.cuda_runtime import (
    CudaStream,
    memcpy_device_to_host_async,
)
from platform.jetson.trt.yolo26_runner import YOLO26PoseTRTRunner


class YOLO26PoseStereoTRTRunner(YOLO26PoseTRTRunner):
    """输入 NCHW float32 [2,3,640,640]；输出 [2,300,57]。

    letterbox 在 trt stream 直写 graph 输入；overlap 后台仅 EGL map。
    """

    def __init__(self, engine_path: Union[str, Path]) -> None:
        super().__init__(engine_path)
        if self.batch_size < 2:
            raise RuntimeError(
                f"Stereo TRT 需要 batch>=2，收到 batch={self.batch_size}: {engine_path}"
            )
        self._letterbox_aux_stream = CudaStream()
        graph_note = (
            "cuda_graph"
            if self._infer_graph is not None and self._infer_graph.valid
            else "enqueue_v3"
        )
        print(
            f"YOLO26PoseStereoTRT: letterbox→trt_input @ trt stream "
            f"({graph_note})"
        )

    def _run_infer_enqueue(self, stream: int) -> None:
        if self._infer_graph is not None and self._infer_graph.valid:
            self._infer_graph.launch()
        else:
            self._enqueue_infer(stream)

    def infer_trt_only(self, *, input_slot: int = 0) -> np.ndarray:
        _ = input_slot
        t1 = time.perf_counter()
        stream = self.stream.handle
        self._run_infer_enqueue(stream)
        t_enqueue = time.perf_counter()
        out_host = self._host_bufs[self.output_name].array
        memcpy_device_to_host_async(
            out_host,
            self._device_bufs[self.output_name].ptr,
            out_host.nbytes,
            stream,
        )
        self.stream.sync()
        t_end = time.perf_counter()
        self._last_infer_enqueue_ms = (t_enqueue - t1) * 1000.0
        self._last_infer_d2h_sync_ms = (t_end - t_enqueue) * 1000.0
        self._last_infer_ms = (t_end - t1) * 1000.0
        return out_host.reshape(self._output_shape)

    @property
    def last_infer_enqueue_ms(self) -> float:
        return float(getattr(self, "_last_infer_enqueue_ms", 0.0))

    @property
    def last_infer_d2h_sync_ms(self) -> float:
        return float(getattr(self, "_last_infer_d2h_sync_ms", 0.0))
