"""OpenCap LSTM marker-enhancer（20 → 解剖 marker），不依赖 Pose2Sim API。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.opencap.opencap20_layout import (
    OPENCAP20_MARKER_NAMES,
    OPENCAP_LSTM_LOWER_FEATURES,
    OPENCAP_LSTM_LOWER_RESPONSE,
    OPENCAP_LSTM_UPPER_FEATURES,
    OPENCAP_LSTM_UPPER_RESPONSE,
)
from core.opencap.subject_profile import SubjectProfile
from core.opencap.trc_io import TrcData, read_trc, write_trc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUGMENTER_DIR = (
    ROOT
    / "Pose2Sim_Full_Pipeline"
    / "pose2sim_src"
    / "Pose2Sim"
    / "MarkerAugmenter"
    / "LSTM"
)


@dataclass(frozen=True)
class EnhanceResult:
    input_trc: Path
    output_trc: Path
    subject_height: float
    subject_mass: float
    num_markers_out: int


def _estimate_height_m(trc: TrcData, default: float = 1.7) -> float:
    """粗略身高：踝→膝→髋→肩链长 + 足跟 + 头近似。"""
    names = {n: i for i, n in enumerate(trc.marker_names)}
    need = ("RAnkle", "RKnee", "RHip", "RShoulder", "LAnkle", "LKnee", "LHip", "LShoulder")
    if any(n not in names for n in need):
        return float(default)

    def _seg(a: str, b: str) -> np.ndarray:
        ia, ib = names[a], names[b]
        d = np.linalg.norm(trc.coords_yup[:, ia] - trc.coords_yup[:, ib], axis=1)
        return d[np.isfinite(d)]

    parts: list[float] = []
    for side in ("R", "L"):
        for a, b in (
            (f"{side}Ankle", f"{side}Knee"),
            (f"{side}Knee", f"{side}Hip"),
            (f"{side}Hip", f"{side}Shoulder"),
        ):
            d = _seg(a, b)
            if d.size:
                parts.append(float(np.median(d)))
    foot = 0.1
    for side in ("R", "L"):
        if f"{side}Heel" in names and f"{side}Ankle" in names:
            d = _seg(f"{side}Heel", f"{side}Ankle")
            if d.size:
                foot = float(np.median(d))
                break
    if len(parts) < 3:
        return float(default)
    # 左右链平均 + foot + 头≈0.25m
    chain = float(np.mean(parts[:3]) + np.mean(parts[3:6])) if len(parts) >= 6 else float(np.mean(parts)) * 2
    # parts already has 6 segments if both sides; mean of (rfoot+lfoot)/2 style:
    if len(parts) >= 6:
        height = (
            foot
            + 0.5 * (parts[0] + parts[3])
            + 0.5 * (parts[1] + parts[4])
            + 0.5 * (parts[2] + parts[5])
            + 0.28
        )
    else:
        height = foot + chain + 0.28
    if not np.isfinite(height) or height < 1.2 or height > 2.3:
        return float(default)
    return float(height)


def _marker_xyz(trc: TrcData, name: str) -> np.ndarray:
    """(T,3)；缺少则 NaN。"""
    if name == "Hip" and "Hip" not in trc.marker_names and "midHip" in trc.marker_names:
        name = "midHip"
    if name == "midHip" and "midHip" not in trc.marker_names and "Hip" in trc.marker_names:
        name = "Hip"
    if name not in trc.marker_names:
        return np.full((trc.coords_yup.shape[0], 3), np.nan, dtype=np.float64)
    i = trc.marker_names.index(name)
    return np.asarray(trc.coords_yup[:, i, :], dtype=np.float64)


def _stack_features(trc: TrcData, feature_names: tuple[str, ...]) -> np.ndarray:
    """(T, 3*F)"""
    cols = [_marker_xyz(trc, n) for n in feature_names]
    return np.concatenate(cols, axis=1)


def _import_onnxruntime():
    """避免项目根目录 platform/ 遮蔽 stdlib.platform（onnxruntime 依赖）。"""
    import importlib
    import sys

    # cwd 在 sys.path[0] 时会优先命中 ./platform
    cwd = ""
    if cwd in sys.path:
        sys.path.remove(cwd)
    plat = sys.modules.get("platform")
    if plat is not None and not hasattr(plat, "system"):
        del sys.modules["platform"]
        # 连带清掉错误的子模块缓存
        for key in list(sys.modules):
            if key == "platform" or key.startswith("platform."):
                del sys.modules[key]
    try:
        return importlib.import_module("onnxruntime")
    except ImportError as exc:
        raise RuntimeError("需要 onnxruntime：pip install onnxruntime") from exc


def _run_one_augmenter(
    *,
    model_dir: Path,
    features: np.ndarray,
    root_xyz: np.ndarray,
    subject_height: float,
    subject_mass: float,
) -> np.ndarray:
    """features (T,3F), root (T,3) → response (T,3R) 世界坐标 Y-up。"""
    ort = _import_onnxruntime()

    mean = np.load(model_dir / "mean.npy", allow_pickle=True)
    std = np.load(model_dir / "std.npy", allow_pickle=True)
    onnx_path = model_dir / "model.onnx"
    if not onnx_path.is_file():
        raise FileNotFoundError(f"缺少 LSTM: {onnx_path}")

    # 相对 midHip，再除以身高
    n_feat_xyz = features.shape[1]
    norm = features - np.tile(root_xyz, (1, n_feat_xyz // 3))
    norm = norm / float(subject_height)
    inputs = np.concatenate(
        [
            norm,
            np.full((norm.shape[0], 1), subject_height, dtype=np.float64),
            np.full((norm.shape[0], 1), subject_mass, dtype=np.float64),
        ],
        axis=1,
    )
    inputs = (inputs - mean) / std
    inputs = np.reshape(inputs, (1, inputs.shape[0], inputs.shape[1])).astype(np.float32)

    session = ort.InferenceSession(str(onnx_path))
    outputs = session.run(["output_0"], {"inputs": inputs})[0]
    outputs = np.reshape(outputs, (outputs.shape[1], outputs.shape[2]))
    unnorm = outputs * float(subject_height)
    unnorm = unnorm + np.tile(root_xyz, (1, unnorm.shape[1] // 3))
    return unnorm


def enhance_trc(
    input_trc: Path,
    output_trc: Path | None = None,
    *,
    subject: SubjectProfile | None = None,
    subject_height: float | None = None,
    subject_mass: float | None = None,
    augmenter_dir: Path | None = None,
    default_height: float = 1.7,
) -> EnhanceResult:
    """
    PreAugmentation TRC（含 OpenCap-20 / midHip）→ 增广解剖 marker TRC。

    身体参数优先用 ``subject``；否则回退 subject_height/mass；再否则估高。
    """
    input_trc = Path(input_trc).expanduser().resolve()
    trc = read_trc(input_trc)
    aug_root = Path(augmenter_dir or DEFAULT_AUGMENTER_DIR).expanduser().resolve()

    if subject is not None:
        height = float(subject.height_m)
        mass = float(subject.mass_kg)
    elif subject_height is not None and float(subject_height) > 0:
        height = float(subject_height)
        mass = float(subject_mass) if subject_mass is not None and float(subject_mass) > 0 else 70.0
    else:
        height = _estimate_height_m(trc, default=default_height)
        mass = float(subject_mass) if subject_mass is not None and float(subject_mass) > 0 else 70.0

    root = _marker_xyz(trc, "midHip")
    if not np.isfinite(root).any():
        raise ValueError("TRC 缺少 midHip/Hip，无法做 OpenCap LSTM 归一化")

    # 校验 feature
    present = set(trc.marker_names)
    present |= {"Hip"} if "midHip" in present else set()
    present |= {"midHip"} if "Hip" in present else set()
    needed = set(OPENCAP_LSTM_LOWER_FEATURES) | set(OPENCAP_LSTM_UPPER_FEATURES)
    missing = sorted(needed - present)
    if missing:
        raise ValueError(f"OpenCap LSTM 缺少 feature markers: {missing}")

    lower_feat = _stack_features(trc, OPENCAP_LSTM_LOWER_FEATURES)
    upper_feat = _stack_features(trc, OPENCAP_LSTM_UPPER_FEATURES)

    if np.isnan(lower_feat).any() or np.isnan(upper_feat).any() or np.isnan(root).any():
        # 用前向填充简单处理（LSTM 对 NaN 敏感）
        for arr in (lower_feat, upper_feat, root):
            _ffill_nan(arr)

    lower_out = _run_one_augmenter(
        model_dir=aug_root / "v0.3_lower",
        features=lower_feat,
        root_xyz=root,
        subject_height=height,
        subject_mass=mass,
    )
    upper_out = _run_one_augmenter(
        model_dir=aug_root / "v0.3_upper",
        features=upper_feat,
        root_xyz=root,
        subject_height=height,
        subject_mass=mass,
    )

    # 组装：先保留输入 20 点顺序（OpenCap20），再拼 response
    base_names = list(OPENCAP20_MARKER_NAMES)
    # 若 TRC 顺序不同，按名取
    t = trc.coords_yup.shape[0]
    base_xyz = np.full((t, len(base_names), 3), np.nan, dtype=np.float64)
    for i, n in enumerate(base_names):
        base_xyz[:, i, :] = _marker_xyz(trc, n)

    resp_names = list(OPENCAP_LSTM_LOWER_RESPONSE) + list(OPENCAP_LSTM_UPPER_RESPONSE)
    resp_xyz = np.concatenate(
        [
            lower_out.reshape(t, len(OPENCAP_LSTM_LOWER_RESPONSE), 3),
            upper_out.reshape(t, len(OPENCAP_LSTM_UPPER_RESPONSE), 3),
        ],
        axis=1,
    )

    all_names = base_names + resp_names
    all_xyz = np.concatenate([base_xyz, resp_xyz], axis=1)

    if output_trc is None:
        output_trc = input_trc.with_name(input_trc.stem + "_OpenCapAug.trc")
    else:
        output_trc = Path(output_trc).expanduser().resolve()

    write_trc(
        output_trc,
        marker_names=all_names,
        frames=trc.frames,
        coords_yup=all_xyz,
        frame_rate=trc.frame_rate,
        units=trc.units,
    )
    return EnhanceResult(
        input_trc=input_trc,
        output_trc=output_trc,
        subject_height=height,
        subject_mass=mass,
        num_markers_out=len(all_names),
    )


def _ffill_nan(arr: np.ndarray) -> None:
    """原地按时间维前向填充 NaN（最后一维是 xyz 或 flat）。"""
    a = arr
    if a.ndim == 1:
        a = a.reshape(-1, 1)
        squeeze = True
    else:
        squeeze = False
    for c in range(a.shape[1]):
        col = a[:, c]
        last = np.nan
        for i in range(col.shape[0]):
            if np.isfinite(col[i]):
                last = col[i]
            elif np.isfinite(last):
                col[i] = last
        # backfill leading
        first = np.nan
        for i in range(col.shape[0] - 1, -1, -1):
            if np.isfinite(col[i]):
                first = col[i]
            elif np.isfinite(first):
                col[i] = first
        a[:, c] = col
    if squeeze:
        arr[:] = a.ravel()
