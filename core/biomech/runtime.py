"""BioModule v1 推理运行时。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from core.biomech.model_v1 import BINARY_CRITERIA, CRITERIA_DIMS, BioModuleV1

DEFAULT_WIN = 27
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "models"
    / "biomodule"
    / "checkpoint"
    / "bio_gt"
    / "bio_gt_best.pth"
)
DEFAULT_NORM_STATS = DEFAULT_CHECKPOINT.parent / "norm_stats.json"


@dataclass(frozen=True)
class NormStats:
    criteria: list[str]
    mean: dict[str, float]
    std: dict[str, float]

    @classmethod
    def load(cls, path: Path) -> NormStats:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "crit_mean" in data:
            mean, std = data["crit_mean"], data["crit_std"]
            criteria = list(data.get("criteria") or mean.keys())
        elif "mean" in data:
            mean, std = data["mean"], data["std"]
            criteria = list(mean.keys())
        else:
            raise ValueError(f"未知 norm_stats 格式: {path}")
        return cls(criteria=list(criteria), mean=mean, std=std)


def load_norm_stats(path: Path | None = None) -> NormStats:
    p = Path(path or DEFAULT_NORM_STATS).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f"找不到 norm_stats: {p}")
    return NormStats.load(p)


def load_biomodule_v1(
    checkpoint: Path | None = None,
    *,
    device: str | torch.device = "cpu",
    win: int = DEFAULT_WIN,
) -> BioModuleV1:
    ckpt_path = Path(checkpoint or DEFAULT_CHECKPOINT).expanduser().resolve()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到权重: {ckpt_path}")

    model = BioModuleV1(win=win)
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=True)
    dev = torch.device(device)
    model.to(dev)
    model.eval()
    return model


def _window_indices(t: int, total: int, win: int) -> np.ndarray:
    half = win // 2
    start = t - half
    idx = np.arange(start, start + win, dtype=np.int64)
    return np.clip(idx, 0, max(total - 1, 0))


def build_pose_windows(
    coords: np.ndarray,
    *,
    win: int = DEFAULT_WIN,
) -> np.ndarray:
    """(T,17,3) → (T, W, 17, 3)，边界 clamp。"""
    coords = np.asarray(coords, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[1:] != (17, 3):
        raise ValueError(f"期望 (T,17,3)，得到 {coords.shape}")
    t_total = coords.shape[0]
    if t_total == 0:
        return np.zeros((0, win, 17, 3), dtype=np.float32)
    out = np.zeros((t_total, win, 17, 3), dtype=np.float32)
    for t in range(t_total):
        out[t] = coords[_window_indices(t, t_total, win)]
    return out


def _denorm_criterion(
    name: str,
    values: np.ndarray,
    stats: NormStats,
) -> np.ndarray:
    if name in BINARY_CRITERIA:
        return (1.0 / (1.0 + np.exp(-values))).astype(np.float32)
    mu = float(stats.mean.get(name, 0.0))
    sigma = float(stats.std.get(name, 1.0))
    if abs(sigma) < 1e-6:
        sigma = 1.0
    return (values * sigma + mu).astype(np.float32)


@torch.no_grad()
def infer_session_poses(
    model: BioModuleV1,
    coords: np.ndarray,
    stats: NormStats,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    center_only: bool = True,
) -> list[dict[str, list[float]]]:
    """
    对整段 (T,17,3) coordinate 序列推理。

    返回长度 T 的 list；每项为 {criterion: [dim...]}（反归一化后）。
    """
    dev = torch.device(device)
    windows = build_pose_windows(coords, win=model.win)
    t_total = windows.shape[0]
    if t_total == 0:
        return []

    center_idx = model.win // 2
    out_rows: list[dict[str, list[float]]] = [
        {} for _ in range(t_total)
    ]

    for b0 in range(0, t_total, batch_size):
        b1 = min(b0 + batch_size, t_total)
        batch = torch.from_numpy(windows[b0:b1]).to(dev)
        preds = model(batch)
        for name in stats.criteria:
            if name not in preds:
                continue
            arr = preds[name].detach().cpu().numpy()
            frame_axis = center_idx if center_only else slice(None)
            if center_only:
                chunk = arr[:, center_idx, :]
            else:
                raise NotImplementedError("center_only=False 暂未实现")
            for i, row in enumerate(chunk):
                den = _denorm_criterion(name, row, stats)
                out_rows[b0 + i][name] = den.reshape(-1).tolist()

    return out_rows


def scalar_summary(row: dict[str, list[float]]) -> dict[str, float]:
    """每 criterion 取均值，便于 jsonl 摘要字段。"""
    out: dict[str, float] = {}
    for name, vals in row.items():
        if not vals:
            continue
        out[name] = float(np.mean(np.asarray(vals, dtype=np.float32)))
    return out
