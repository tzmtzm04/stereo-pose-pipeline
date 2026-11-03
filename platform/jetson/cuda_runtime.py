"""Minimal CUDA runtime bindings via ctypes (no PyTorch)."""

from __future__ import annotations

import ctypes
from ctypes import c_void_p, byref, c_size_t

import numpy as np

_cuda = ctypes.CDLL("libcudart.so.12")

cudaMemcpyHostToDevice = 1
cudaMemcpyDeviceToHost = 2
cudaMemcpyDeviceToDevice = 3
cudaStreamCaptureModeGlobal = 0

_cuda.cudaMemcpyAsync.argtypes = [
    c_void_p,
    c_void_p,
    c_size_t,
    ctypes.c_int,
    c_void_p,
]
_cuda.cudaMemcpyAsync.restype = ctypes.c_int

_cuda.cudaMemsetAsync.argtypes = [c_void_p, ctypes.c_uint, c_size_t, c_void_p]
_cuda.cudaMemsetAsync.restype = ctypes.c_int

_cuda.cudaStreamBeginCapture.argtypes = [c_void_p, ctypes.c_int]
_cuda.cudaStreamBeginCapture.restype = ctypes.c_int

_cuda.cudaStreamEndCapture.argtypes = [c_void_p, ctypes.POINTER(c_void_p)]
_cuda.cudaStreamEndCapture.restype = ctypes.c_int

_cuda.cudaGraphInstantiate.argtypes = [
    ctypes.POINTER(c_void_p),
    c_void_p,
    ctypes.c_ulonglong,
]
_cuda.cudaGraphInstantiate.restype = ctypes.c_int

_cuda.cudaGraphLaunch.argtypes = [c_void_p, c_void_p]
_cuda.cudaGraphLaunch.restype = ctypes.c_int

_cuda.cudaGraphExecDestroy.argtypes = [c_void_p]
_cuda.cudaGraphExecDestroy.restype = ctypes.c_int

_cuda.cudaGraphDestroy.argtypes = [c_void_p]
_cuda.cudaGraphDestroy.restype = ctypes.c_int

_cuda.cudaEventCreate.argtypes = [ctypes.POINTER(c_void_p), ctypes.c_uint]
_cuda.cudaEventCreate.restype = ctypes.c_int

_cuda.cudaEventRecord.argtypes = [c_void_p, c_void_p]
_cuda.cudaEventRecord.restype = ctypes.c_int

_cuda.cudaStreamWaitEvent.argtypes = [c_void_p, c_void_p, ctypes.c_uint]
_cuda.cudaStreamWaitEvent.restype = ctypes.c_int

_cuda.cudaEventDestroy.argtypes = [c_void_p]
_cuda.cudaEventDestroy.restype = ctypes.c_int

cudaEventDisableTiming = 1


def _check(err: int, what: str) -> None:
    if err != 0:
        raise RuntimeError(f"CUDA error {err} in {what}")


def sync_stream(stream_handle: int) -> None:
    _check(
        _cuda.cudaStreamSynchronize(c_void_p(stream_handle)),
        "cudaStreamSynchronize",
    )


class CudaStream:
    def __init__(self) -> None:
        self._handle = c_void_p()
        _check(_cuda.cudaSetDevice(0), "cudaSetDevice")
        _check(_cuda.cudaStreamCreate(byref(self._handle)), "cudaStreamCreate")

    @property
    def handle(self) -> int:
        return self._handle.value or 0

    def sync(self) -> None:
        _check(_cuda.cudaStreamSynchronize(self._handle), "cudaStreamSynchronize")


class CudaEvent:
    """跨 stream 依赖：prep_stream record → trt_stream wait。"""

    def __init__(self) -> None:
        self._handle = c_void_p()
        _check(
            _cuda.cudaEventCreate(byref(self._handle), cudaEventDisableTiming),
            "cudaEventCreate",
        )

    @property
    def handle(self) -> int:
        return self._handle.value or 0

    def record(self, stream: CudaStream | int) -> None:
        sh = stream.handle if isinstance(stream, CudaStream) else int(stream)
        _check(_cuda.cudaEventRecord(self._handle, c_void_p(sh)), "cudaEventRecord")

    def stream_wait(self, stream: CudaStream | int, *, flags: int = 0) -> None:
        sh = stream.handle if isinstance(stream, CudaStream) else int(stream)
        _check(
            _cuda.cudaStreamWaitEvent(
                c_void_p(sh),
                self._handle,
                ctypes.c_uint(flags),
            ),
            "cudaStreamWaitEvent",
        )

    def destroy(self) -> None:
        if self._handle.value:
            _cuda.cudaEventDestroy(self._handle)
            self._handle = c_void_p()


class CudaBuffer:
    def __init__(self, nbytes: int) -> None:
        self.nbytes = nbytes
        self._ptr = c_void_p()
        _check(_cuda.cudaMalloc(byref(self._ptr), nbytes), "cudaMalloc")

    @property
    def ptr(self) -> int:
        return self._ptr.value or 0

    def free(self) -> None:
        if self._ptr.value:
            _check(_cuda.cudaFree(self._ptr), "cudaFree")
            self._ptr = c_void_p()

    def copy_from_host(self, arr) -> None:
        host = arr.ctypes.data_as(c_void_p)
        _check(
            _cuda.cudaMemcpy(self._ptr, host, self.nbytes, cudaMemcpyHostToDevice),
            "cudaMemcpy H2D",
        )

    def copy_to_host(self, arr) -> None:
        host = arr.ctypes.data_as(c_void_p)
        _check(
            _cuda.cudaMemcpy(host, self._ptr, self.nbytes, cudaMemcpyDeviceToHost),
            "cudaMemcpy D2H",
        )


class PinnedHostArray:
    """Pinned host memory（page-locked），供 cudaMemcpyAsync 使用。"""

    def __init__(self, count: int, dtype: np.dtype | type) -> None:
        self.dtype = np.dtype(dtype)
        self.count = int(count)
        self.nbytes = self.count * self.dtype.itemsize
        self._ptr = c_void_p()
        _check(
            _cuda.cudaMallocHost(byref(self._ptr), c_size_t(self.nbytes)),
            "cudaMallocHost",
        )
        byte_arr = (ctypes.c_byte * self.nbytes).from_address(self._ptr.value or 0)
        self.array = np.frombuffer(byte_arr, dtype=self.dtype, count=self.count)

    @property
    def ptr(self) -> int:
        return self._ptr.value or 0

    def free(self) -> None:
        if self._ptr.value:
            _check(_cuda.cudaFreeHost(self._ptr), "cudaFreeHost")
            self._ptr = c_void_p()
            self.array = np.empty(0, dtype=self.dtype)


def memcpy_host_to_device(dst_ptr: int, src_arr) -> None:
    host = src_arr.ctypes.data_as(c_void_p)
    nbytes = c_size_t(src_arr.nbytes)
    _check(
        _cuda.cudaMemcpy(c_void_p(dst_ptr), host, nbytes, cudaMemcpyHostToDevice),
        "cudaMemcpy H2D",
    )


def memcpy_device_to_host(dst_arr, src_ptr: int) -> None:
    host = dst_arr.ctypes.data_as(c_void_p)
    nbytes = c_size_t(dst_arr.nbytes)
    _check(
        _cuda.cudaMemcpy(host, c_void_p(src_ptr), nbytes, cudaMemcpyDeviceToHost),
        "cudaMemcpy D2H",
    )


def memcpy_host_to_device_async(
    dst_ptr: int,
    src_arr,
    nbytes: int,
    stream_handle: int,
) -> None:
    host = src_arr.ctypes.data_as(c_void_p)
    _check(
        _cuda.cudaMemcpyAsync(
            c_void_p(dst_ptr),
            host,
            c_size_t(nbytes),
            cudaMemcpyHostToDevice,
            c_void_p(stream_handle),
        ),
        "cudaMemcpyAsync H2D",
    )


def memcpy_device_to_host_async(
    dst_arr,
    src_ptr: int,
    nbytes: int,
    stream_handle: int,
) -> None:
    host = dst_arr.ctypes.data_as(c_void_p)
    _check(
        _cuda.cudaMemcpyAsync(
            host,
            c_void_p(src_ptr),
            c_size_t(nbytes),
            cudaMemcpyDeviceToHost,
            c_void_p(stream_handle),
        ),
        "cudaMemcpyAsync D2H",
    )


def memcpy_device_to_device_async(
    dst_ptr: int,
    src_ptr: int,
    nbytes: int,
    stream_handle: int,
) -> None:
    _check(
        _cuda.cudaMemcpyAsync(
            c_void_p(dst_ptr),
            c_void_p(src_ptr),
            c_size_t(nbytes),
            cudaMemcpyDeviceToDevice,
            c_void_p(stream_handle),
        ),
        "cudaMemcpyAsync D2D",
    )


class CudaGraphExec:
    """捕获并在固定 stream 上重放 enqueue（如 TRT execute_async_v3）。"""

    def __init__(self, stream: CudaStream) -> None:
        self._stream = stream
        self._graph = c_void_p()
        self._graph_exec = c_void_p()
        self._valid = False

    @property
    def valid(self) -> bool:
        return self._valid

    def capture_enqueue(self, enqueue_fn) -> bool:
        """enqueue_fn 在 stream 上提交 GPU 工作，不得 sync。"""
        sh = self._stream._handle
        _check(
            _cuda.cudaStreamBeginCapture(sh, cudaStreamCaptureModeGlobal),
            "cudaStreamBeginCapture",
        )
        enqueue_fn()
        err = _cuda.cudaStreamEndCapture(sh, byref(self._graph))
        if err != 0:
            self._valid = False
            return False
        err = _cuda.cudaGraphInstantiate(
            byref(self._graph_exec),
            self._graph,
            ctypes.c_ulonglong(0),
        )
        self._valid = err == 0
        return self._valid

    def launch(self) -> None:
        if not self._valid:
            raise RuntimeError("CudaGraphExec not captured")
        _check(
            _cuda.cudaGraphLaunch(self._graph_exec, self._stream._handle),
            "cudaGraphLaunch",
        )

    def destroy(self) -> None:
        if self._graph_exec.value:
            _cuda.cudaGraphExecDestroy(self._graph_exec)
            self._graph_exec = c_void_p()
        if self._graph.value:
            _cuda.cudaGraphDestroy(self._graph)
            self._graph = c_void_p()
        self._valid = False


def memset_device_async(ptr: int, nbytes: int, stream_handle: int) -> None:
    _check(
        _cuda.cudaMemsetAsync(
            c_void_p(ptr),
            0,
            c_size_t(nbytes),
            c_void_p(stream_handle),
        ),
        "cudaMemsetAsync",
    )
