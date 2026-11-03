"""SmoothNet FCN（numpy 推理，无 PyTorch 运行时依赖）。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _leaky_relu(x: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    return np.where(x > 0, x, alpha * x).astype(np.float32, copy=False)


def _linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    # x: (..., in), w: (out, in), b: (out,)
    return (x @ w.T + b).astype(np.float32, copy=False)


def _res_block(x: np.ndarray, weights: dict, prefix: str) -> np.ndarray:
    identity = x
    h = _linear(x, weights[f"{prefix}.linear1.weight"], weights[f"{prefix}.linear1.bias"])
    h = _leaky_relu(h, alpha=0.2)
    h = _linear(h, weights[f"{prefix}.linear2.weight"], weights[f"{prefix}.linear2.bias"])
    h = _leaky_relu(h, alpha=0.2)
    return (h + identity).astype(np.float32, copy=False)


@dataclass(frozen=True)
class SmoothNetConfig:
    window_size: int
    output_size: int
    pose_dim: int
    hidden_size: int = 512
    res_hidden_size: int = 256
    num_blocks: int = 3


class SmoothNetNumpy:
    """输入 (1, C, T)，输出 (1, C, output_size)。"""

    def __init__(self, cfg: SmoothNetConfig, weights: dict[str, np.ndarray]) -> None:
        self.cfg = cfg
        self._w = weights

    @classmethod
    def load_npz(cls, path: Path | str) -> "SmoothNetNumpy":
        p = Path(path).expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"SmoothNet npz 不存在: {p}")
        data = np.load(p, allow_pickle=False)
        meta = {k: int(data[k]) for k in data.files if k.startswith("meta_")}
        cfg = SmoothNetConfig(
            window_size=int(meta.get("meta_window_size", data["meta_window_size"])),
            output_size=int(meta.get("meta_output_size", data["meta_output_size"])),
            pose_dim=int(meta.get("meta_pose_dim", data["meta_pose_dim"])),
            hidden_size=int(meta.get("meta_hidden_size", 512)),
            res_hidden_size=int(meta.get("meta_res_hidden_size", 256)),
            num_blocks=int(meta.get("meta_num_blocks", 3)),
        )
        weights = {k: np.asarray(data[k]) for k in data.files if not k.startswith("meta_")}
        return cls(cfg, weights)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """x: (1, C, T) float32 → (1, C, output_size)."""
        cfg = self.cfg
        if x.shape != (1, cfg.pose_dim, cfg.window_size):
            raise ValueError(
                f"SmoothNet 输入应为 (1, {cfg.pose_dim}, {cfg.window_size})，收到 {x.shape}"
            )
        w = self._w
        h = _linear(x, w["encoder.0.weight"], w["encoder.0.bias"])
        h = _leaky_relu(h, alpha=0.1)
        for i in range(cfg.num_blocks):
            h = _res_block(h, w, f"res_blocks.{i}")
        out = _linear(h, w["decoder.weight"], w["decoder.bias"])
        return out.astype(np.float32, copy=False)
