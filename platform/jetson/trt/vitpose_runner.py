"""ViTPose++ TensorRT runner（heatmap top-down，无 PyTorch）。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Union

import numpy as np
import tensorrt as trt

from app.paths import (
    DEFAULT_VITPOSE_PP_ENGINE,
    DEFAULT_VITPOSE_PP_L_COCO_ENGINE_FP16,
    DEFAULT_VITPOSE_PP_L_COCO_ENGINE_FP32,
    resolve_vitpose_pp_l_coco_engine,
)
from core.offline.two_stage.vitpose_input import (
    VITPOSE_INPUT_H,
    VITPOSE_INPUT_W,
    VITPOSE_NCHW_SHAPE,
    assert_bgr_matches_vitpose,
)
from core.offline.two_stage.vitpose_postprocess import (
    heatmaps_to_keypoints,
    heatmaps_to_keypoints_hypotheses,
    infer_num_keypoints_from_heatmaps,
)
from platform.jetson.cuda_runtime import (
    CudaBuffer,
    CudaGraphExec,
    CudaStream,
    PinnedHostArray,
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
    memset_device_async,
)

_IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class ViTPoseTRTRunner:
    """输入 NCHW float32 RGB，mmpose W×H=192×256（H×W=256×192）；输出 heatmap [1,K,H,W]（K=17 或 133）。"""

    gpu_egl_capable = True

    def __init__(
        self,
        engine_path: Union[str, Path] = DEFAULT_VITPOSE_PP_ENGINE,
    ) -> None:
        resolved = Path(engine_path).expanduser().resolve()
        if not resolved.is_file() and resolved == DEFAULT_VITPOSE_PP_L_COCO_ENGINE_FP32.resolve():
            resolved = resolve_vitpose_pp_l_coco_engine(prefer_fp32=True)
        self.engine_path = resolved
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"ViTPose engine not found: {self.engine_path}")

        self.logger = trt.Logger(trt.Logger.WARNING)
        with self.engine_path.open("rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = CudaStream()
        self.input_name = ""
        self.output_name = ""
        self.input_h = VITPOSE_INPUT_H
        self.input_w = VITPOSE_INPUT_W
        self.batch_size = 1
        self._device_bufs: dict[str, CudaBuffer] = {}
        self._host_bufs: dict[str, PinnedHostArray] = {}
        self._detect_io()
        if (self.input_h, self.input_w) != (VITPOSE_INPUT_H, VITPOSE_INPUT_W):
            raise RuntimeError(
                f"ViTPose engine 输入 H×W={self.input_h}×{self.input_w} "
                f"!= 约定 {VITPOSE_INPUT_H}×{VITPOSE_INPUT_W}；"
                f"请用匹配 NCHW={list(VITPOSE_NCHW_SHAPE)} 的 engine"
            )
        self._alloc_buffers()
        self._output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self.num_keypoints = infer_num_keypoints_from_heatmaps(
            np.empty(self._output_shape, dtype=np.float32)
        )
        self._bind_tensor_addresses()
        self._infer_graph: CudaGraphExec | None = None
        self._warmup_and_capture_infer_graph()
        graph_note = (
            "cuda_graph" if self._infer_graph and self._infer_graph.valid else "enqueue_v3"
        )
        print(
            f"ViTPoseTRT: {self.engine_path.name} GPU W×H={self.input_w}x{self.input_h} "
            f"out={self._output_shape} kpts={self.num_keypoints} ({graph_note})",
            flush=True,
        )

    def _enqueue_infer(self, stream: int) -> None:
        self.context.execute_async_v3(stream)

    def _warmup_and_capture_infer_graph(self, warmup_runs: int = 3) -> None:
        stream = self.stream.handle
        inp = self._device_bufs[self.input_name]
        memset_device_async(inp.ptr, inp.nbytes, stream)
        for _ in range(warmup_runs):
            self._enqueue_infer(stream)
        out_host = self._host_bufs[self.output_name].array
        memcpy_device_to_host_async(
            out_host,
            self._device_bufs[self.output_name].ptr,
            out_host.nbytes,
            stream,
        )
        self.stream.sync()
        graph = CudaGraphExec(self.stream)
        if graph.capture_enqueue(lambda: self._enqueue_infer(stream)):
            self._infer_graph = graph
        else:
            graph.destroy()

    def _detect_io(self) -> None:
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_name = name
                if len(shape) == 4:
                    self.batch_size = max(1, int(shape[0]))
                    self.input_h = int(shape[2])
                    self.input_w = int(shape[3])
            else:
                self.output_name = name

    def _alloc_buffers(self) -> None:
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = int(np.prod(shape))
            self._host_bufs[name] = PinnedHostArray(size, dtype)
            self._device_bufs[name] = CudaBuffer(size * np.dtype(dtype).itemsize)

    def _bind_tensor_addresses(self) -> None:
        self.context.set_tensor_address(
            self.input_name, self._device_bufs[self.input_name].ptr
        )
        for name, buf in self._device_bufs.items():
            if name != self.input_name:
                self.context.set_tensor_address(name, buf.ptr)

    def _preprocess_crop_bgr(self, crop_bgr: np.ndarray) -> np.ndarray:
        assert_bgr_matches_vitpose(crop_bgr.shape)
        rgb = crop_bgr[:, :, ::-1].astype(np.float32)
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        return rgb.transpose(2, 0, 1)

    def infer_blob(self, blob_nchw: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        stream = self.stream.handle
        inp_host = self._host_bufs[self.input_name].array
        flat = blob_nchw.ravel()
        inp_host[: flat.size] = flat
        memcpy_host_to_device_async(
            self._device_bufs[self.input_name].ptr, inp_host, flat.nbytes, stream
        )
        if self._infer_graph is not None and self._infer_graph.valid:
            self._infer_graph.launch()
        else:
            self._enqueue_infer(stream)
        out_host = self._host_bufs[self.output_name].array
        memcpy_device_to_host_async(
            out_host,
            self._device_bufs[self.output_name].ptr,
            out_host.nbytes,
            stream,
        )
        self.stream.sync()
        self._last_infer_ms = (time.perf_counter() - t0) * 1000.0
        return out_host.reshape(self._output_shape)

    def infer_crop_bgr(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.infer_blob(self._preprocess_crop_bgr(crop_bgr))

    def keypoints_from_heatmaps(self, heatmaps: np.ndarray) -> np.ndarray:
        return heatmaps_to_keypoints(
            heatmaps,
            input_w=self.input_w,
            input_h=self.input_h,
            num_keypoints=self.num_keypoints,
        )

    def keypoints_hypo_from_heatmaps(
        self,
        heatmaps: np.ndarray,
        *,
        top_p: int = 3,
        min_dist: float = 1.0,
        score_min: float = 0.05,
        peak_decode_mode: str = "local",
        local_radius: int = 1,
        score_ratio: float = 0.6,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        top_p = int(top_p)
        if top_p <= 1:
            return self.keypoints_from_heatmaps(heatmaps), None
        hypo = heatmaps_to_keypoints_hypotheses(
            heatmaps,
            top_p=top_p,
            min_dist=float(min_dist),
            score_min=float(score_min),
            input_w=self.input_w,
            input_h=self.input_h,
            num_keypoints=self.num_keypoints,
            peak_decode_mode=str(peak_decode_mode),
            local_radius=int(local_radius),
            score_ratio=float(score_ratio),
        )
        return hypo[:, 0, :].copy(), hypo

    def infer_heatmaps_from_crop_bgr(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.infer_crop_bgr(crop_bgr)

    def keypoints_from_crop_bgr(self, crop_bgr: np.ndarray) -> np.ndarray:
        heatmaps = self.infer_crop_bgr(crop_bgr)
        return self.keypoints_from_heatmaps(heatmaps)

    def infer_trt_only(self) -> np.ndarray:
        t1 = time.perf_counter()
        stream = self.stream.handle
        if self._infer_graph is not None and self._infer_graph.valid:
            self._infer_graph.launch()
        else:
            self._enqueue_infer(stream)
        out_host = self._host_bufs[self.output_name].array
        memcpy_device_to_host_async(
            out_host,
            self._device_bufs[self.output_name].ptr,
            out_host.nbytes,
            stream,
        )
        self.stream.sync()
        self._last_infer_ms = (time.perf_counter() - t1) * 1000.0
        return out_host.reshape(self._output_shape)

    @property
    def input_device_ptr(self) -> int:
        return int(self._device_bufs[self.input_name].ptr)

    @property
    def last_infer_ms(self) -> float:
        return getattr(self, "_last_infer_ms", 0.0)
