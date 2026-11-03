"""YOLO26n-pose TensorRT runner（ctypes CUDA，无 PyTorch）。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Union

import numpy as np
import tensorrt as trt

from app.paths import DEFAULT_YOLO26_POSE_ENGINE
from platform.jetson.cuda.nv12_rgb import cuda_nv12_rgb_available, nv12_er_pitch_to_nchw_rgb
from platform.jetson.cuda_runtime import (
    CudaBuffer,
    CudaGraphExec,
    CudaStream,
    PinnedHostArray,
    memcpy_device_to_host_async,
    memcpy_host_to_device_async,
    memset_device_async,
)
from platform.jetson.nvbuf_egl_cuda import Nv12CudaPitchPlanes

_egl_nv12_feed_lock = threading.Lock()


class YOLO26PoseTRTRunner:
    """输入 NCHW float32 RGB 640×640；输出 V26 [1,300,57]。"""

    def __init__(
        self,
        engine_path: Union[str, Path] = DEFAULT_YOLO26_POSE_ENGINE,
        *,
        dla_core: int | None = None,
    ) -> None:
        self.engine_path = Path(engine_path).expanduser().resolve()
        if not self.engine_path.is_file():
            raise FileNotFoundError(f"YOLO26 pose engine not found: {self.engine_path}")

        self._dla_core: int | None = int(dla_core) if dla_core is not None else None
        self.logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(self.logger)
        if self._dla_core is not None:
            runtime.DLA_core = self._dla_core
        with self.engine_path.open("rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load engine: {self.engine_path}")

        self.context = self.engine.create_execution_context()
        self.stream = CudaStream()
        self.input_name = ""
        self.output_name = ""
        self.input_h = 640
        self.input_w = 640
        self.batch_size = 1
        self._device_bufs: dict[str, CudaBuffer] = {}
        self._host_bufs: dict[str, PinnedHostArray] = {}
        self._detect_io()
        self._input_batch_stride = int(
            3 * self.input_w * self.input_h * np.dtype(np.float32).itemsize
        )
        self._alloc_buffers()
        self._output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
        self._bind_tensor_addresses()
        self._infer_graph: CudaGraphExec | None = None
        if self._dla_core is None:
            self._warmup_and_capture_infer_graph()
        graph_note = (
            "cuda_graph"
            if self._infer_graph and self._infer_graph.valid
            else "enqueue_v3"
        )
        dev_tag = f"DLA{self._dla_core}" if self._dla_core is not None else "GPU"
        batch_tag = f"batch={self.batch_size} " if self.batch_size > 1 else ""
        print(
            f"YOLO26PoseTRT: {self.engine_path.name} {batch_tag}{dev_tag} "
            f"{self.input_w}x{self.input_h} ({graph_note})"
        )

    def _enqueue_infer(self, stream: int) -> None:
        self.context.execute_async_v3(stream)

    def _warmup_and_capture_infer_graph(self, warmup_runs: int = 10) -> None:
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

    def _preprocess_bgr_letterbox(self, bgr: np.ndarray) -> np.ndarray:
        from platform.jetson.letterbox_util import letterbox_bgr_cpu

        canvas, _ = letterbox_bgr_cpu(bgr, self.input_w, self.input_h)
        rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
        return rgb.transpose(2, 0, 1)

    def _preprocess_bgr_nchw(self, bgr: np.ndarray) -> np.ndarray:
        if bgr.shape[1] != self.input_w or bgr.shape[0] != self.input_h:
            raise ValueError(
                f"期望已 letterbox 的 {self.input_w}x{self.input_h}，收到 {bgr.shape[1]}x{bgr.shape[0]}"
            )
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
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
        self._run_infer_enqueue(stream)
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

    def infer_bgr_letterbox(self, bgr: np.ndarray) -> np.ndarray:
        return self.infer_blob(self._preprocess_bgr_letterbox(bgr))

    def infer_bgr_letterboxed(self, bgr_letterbox: np.ndarray) -> np.ndarray:
        return self.infer_blob(self._preprocess_bgr_nchw(bgr_letterbox))

    def input_device_ptr_at(self, batch_index: int = 0) -> int:
        idx = int(batch_index)
        if idx < 0 or idx >= self.batch_size:
            raise IndexError(
                f"batch_index {idx} out of range [0, {self.batch_size})"
            )
        base_ptr = self._device_bufs[self.input_name].ptr
        return int(base_ptr + idx * self._input_batch_stride)

    def feed_nv12_planes_to_input_at(
        self,
        batch_index: int,
        planes: Nv12CudaPitchPlanes,
    ) -> None:
        if not cuda_nv12_rgb_available():
            raise RuntimeError(
                "route2_nv12_rgb.so 未构建: bash scripts/route2/build_cuda_nv12_rgb.sh"
            )
        w, h, p = planes.width, planes.height, planes.pitch
        if w != self.input_w or h != self.input_h:
            raise ValueError(f"NV12 {w}x{h} != engine {self.input_w}x{self.input_h}")
        from platform.jetson.nvbuf_egl_cuda import get_shared_egl_cuda_mapper

        mapper = get_shared_egl_cuda_mapper()
        stream = self.stream.handle
        dst_ptr = self.input_device_ptr_at(batch_index)
        with _egl_nv12_feed_lock:
            mapped = mapper.map_nv12_planes(planes, stream)
            try:
                t0 = time.perf_counter()
                nv12_er_pitch_to_nchw_rgb(
                    mapped.y_ptr,
                    mapped.uv_ptr,
                    dst_ptr,
                    w,
                    h,
                    p,
                    stream,
                )
                self._preprocess_ms = (time.perf_counter() - t0) * 1000.0
            finally:
                mapper.unmap_nv12_planes(planes, stream)

    def feed_nv12_planes_to_input(self, planes: Nv12CudaPitchPlanes) -> None:
        self.feed_nv12_planes_to_input_at(0, planes)

    def _run_infer_enqueue(self, stream: int) -> None:
        if self._infer_graph is not None and self._infer_graph.valid:
            self._infer_graph.launch()
        else:
            self._enqueue_infer(stream)

    def infer_trt_only(self) -> np.ndarray:
        t1 = time.perf_counter()
        stream = self.stream.handle
        self._run_infer_enqueue(stream)
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

    def infer_nv12_cuda_planes(self, planes: Nv12CudaPitchPlanes) -> np.ndarray:
        self.feed_nv12_planes_to_input(planes)
        return self.infer_trt_only()

    @property
    def last_infer_ms(self) -> float:
        return getattr(self, "_last_infer_ms", 0.0)

    @property
    def last_preprocess_ms(self) -> float:
        return getattr(self, "_preprocess_ms", 0.0)

    @last_preprocess_ms.setter
    def last_preprocess_ms(self, value: float) -> None:
        self._preprocess_ms = float(value)

    @property
    def input_device_ptr(self) -> int:
        return int(self._device_bufs[self.input_name].ptr)

    @property
    def dla_core(self) -> int | None:
        return self._dla_core
