"""DWPose ONNXRuntime runner（SimCC WholeBody 133，256×192）。"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Union

import numpy as np


def _import_onnxruntime():
    """repo 顶层 platform/ 会遮蔽 stdlib platform，ORT 导入前需临时恢复。"""
    proj = sys.modules.get("platform")
    paths = sys.path[:]
    filtered = [
        p for p in paths if not (Path(p).expanduser().resolve() / "platform" / "jetson").is_dir()
    ]
    sys.path[:] = filtered
    if "platform" in sys.modules and not hasattr(sys.modules["platform"], "system"):
        del sys.modules["platform"]
    import onnxruntime as ort

    sys.path[:] = paths
    if proj is not None:
        sys.modules["platform"] = proj
    return ort


ort = _import_onnxruntime()

from app.paths import DEFAULT_DWPOSE_L_ONNX
from core.offline.two_stage.coco_wholebody_layout import NUM_KEYPOINTS_WHOLEBODY
from core.offline.two_stage.dwpose_postprocess import (
    DEFAULT_SIMCC_SPLIT_RATIO,
    simcc_to_keypoints,
)
from core.offline.two_stage.vitpose_input import (
    VITPOSE_INPUT_H,
    VITPOSE_INPUT_W,
    assert_bgr_matches_vitpose,
)

_IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
_IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


class DWPoseOnnxRunner:
    """输入 NCHW RGB ImageNet；输出 simcc_x + simcc_y → (K,3) crop 坐标。"""

    gpu_egl_capable = False

    def __init__(
        self,
        onnx_path: Union[str, Path] = DEFAULT_DWPOSE_L_ONNX,
        *,
        providers: list[str] | None = None,
        simcc_split_ratio: float = DEFAULT_SIMCC_SPLIT_RATIO,
    ) -> None:
        self.onnx_path = Path(onnx_path).expanduser().resolve()
        if not self.onnx_path.is_file():
            raise FileNotFoundError(
                f"DWPose ONNX 不存在: {self.onnx_path}\n"
                "请先: bash scripts/offline/export_dwpose_onnx.sh"
            )
        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self._session = ort.InferenceSession(
            str(self.onnx_path),
            providers=providers,
        )
        self.input_name = self._session.get_inputs()[0].name
        self._output_names = [o.name for o in self._session.get_outputs()]
        in_shape = self._session.get_inputs()[0].shape
        self.input_h = (
            int(in_shape[2]) if in_shape[2] not in (None, "None") else VITPOSE_INPUT_H
        )
        self.input_w = (
            int(in_shape[3]) if in_shape[3] not in (None, "None") else VITPOSE_INPUT_W
        )
        if (self.input_w, self.input_h) != (VITPOSE_INPUT_W, VITPOSE_INPUT_H):
            raise RuntimeError(
                f"DWPose ONNX 输入须为 W×H={VITPOSE_INPUT_W}×{VITPOSE_INPUT_H}，"
                f"收到 {self.input_w}×{self.input_h}；请导出 256×192 模型"
            )
        self.num_keypoints = NUM_KEYPOINTS_WHOLEBODY
        self._simcc_split_ratio = float(simcc_split_ratio)
        self._last_infer_ms = 0.0
        active = self._session.get_providers()
        print(
            f"DWPoseONNX: {self.onnx_path.name} W×H={self.input_w}x{self.input_h} "
            f"kpts={self.num_keypoints} provider={active[0] if active else '?'}",
            flush=True,
        )

    def _preprocess_crop_bgr(self, crop_bgr: np.ndarray) -> np.ndarray:
        assert_bgr_matches_vitpose(crop_bgr.shape)
        rgb = crop_bgr[:, :, ::-1].astype(np.float32)
        rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
        return rgb.transpose(2, 0, 1)[None, ...].astype(np.float32)

    def infer_blob(self, blob_nchw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        t0 = time.perf_counter()
        outs = self._session.run(self._output_names, {self.input_name: blob_nchw})
        self._last_infer_ms = (time.perf_counter() - t0) * 1000.0
        if len(outs) != 2:
            raise RuntimeError(
                f"期望 SimCC 双输出 simcc_x/simcc_y，收到 {len(outs)} 个输出"
            )
        return outs[0], outs[1]

    def keypoints_from_simcc(
        self,
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
    ) -> np.ndarray:
        return simcc_to_keypoints(
            simcc_x,
            simcc_y,
            input_w=self.input_w,
            input_h=self.input_h,
            simcc_split_ratio=self._simcc_split_ratio,
            num_keypoints=self.num_keypoints,
        )

    def keypoints_from_crop_bgr(self, crop_bgr: np.ndarray) -> np.ndarray:
        blob = self._preprocess_crop_bgr(crop_bgr)
        sx, sy = self.infer_blob(blob)
        return self.keypoints_from_simcc(sx, sy)

    @property
    def last_infer_ms(self) -> float:
        return float(self._last_infer_ms)
