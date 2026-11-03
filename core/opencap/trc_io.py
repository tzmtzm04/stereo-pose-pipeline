"""OpenCap / OpenSim 风格 TRC 读写（Units=m，Y-up）。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def left_cam_opencv_to_zup(xyz: np.ndarray) -> np.ndarray:
    """左相机 OpenCV (X右,Y下,Z前) → Z-up 中间系。"""
    pts = np.asarray(xyz, dtype=np.float32)
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    return np.stack([x, z, -y], axis=-1)


def zup_to_yup(flat_zup: np.ndarray) -> np.ndarray:
    """(T, 3*M) Z-up → Y-up（OpenSim / OpenCap）。"""
    out = np.empty_like(flat_zup)
    n_markers = flat_zup.shape[1] // 3
    for i in range(n_markers):
        out[:, i * 3 + 0] = flat_zup[:, i * 3 + 1]
        out[:, i * 3 + 1] = flat_zup[:, i * 3 + 2]
        out[:, i * 3 + 2] = flat_zup[:, i * 3 + 0]
    return out


def xyz_left_cam_to_yup(xyz: np.ndarray) -> np.ndarray:
    """单点或 (...,3) 左相机 → Y-up。"""
    zup = left_cam_opencv_to_zup(xyz)
    pts = np.asarray(zup, dtype=np.float32)
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    return np.stack([y, z, x], axis=-1)


def yup_to_left_cam_opencv(xyz: np.ndarray) -> np.ndarray:
    """Y-up（OpenSim / OpenCap）→ 左相机 OpenCV (X右,Y下,Z前)。xyz_left_cam_to_yup 的逆。"""
    pts = np.asarray(xyz, dtype=np.float32)
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    return np.stack([z, -y, x], axis=-1)


@dataclass
class TrcData:
    marker_names: list[str]
    frames: np.ndarray  # (T,) int
    times: np.ndarray  # (T,) float
    coords_yup: np.ndarray  # (T, M, 3)
    frame_rate: float
    units: str = "m"


def write_trc(
    path: Path,
    *,
    marker_names: tuple[str, ...] | list[str],
    frames: np.ndarray,
    coords_yup: np.ndarray,
    frame_rate: float,
    units: str = "m",
) -> None:
    """
    coords_yup: (T, M, 3) 或 (T, 3*M) Y-up 米制。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    names = list(marker_names)
    frames = np.asarray(frames, dtype=np.int64)
    arr = np.asarray(coords_yup, dtype=np.float64)
    if arr.ndim == 2:
        if arr.shape[1] != len(names) * 3:
            raise ValueError(f"flat coords 列数 {arr.shape[1]} != 3*{len(names)}")
        arr = arr.reshape(arr.shape[0], len(names), 3)
    if arr.ndim != 3 or arr.shape[1] != len(names) or arr.shape[2] != 3:
        raise ValueError(f"期望 coords (T,M,3)，收到 {arr.shape}")

    n_frames = int(arr.shape[0])
    n_markers = len(names)
    start_frame = int(frames[0]) if n_frames else 1
    trc_name = path.name

    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{trc_name}",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames",
        "\t".join(
            str(x)
            for x in (
                frame_rate,
                frame_rate,
                n_frames,
                n_markers,
                units,
                frame_rate,
                start_frame,
                n_frames,
            )
        ),
        "Frame#\tTime\t" + "\t\t\t".join(names) + "\t\t\t",
        "\t\t"
        + "\t".join(f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(n_markers))
        + "\t",
    ]

    with path.open("w", encoding="utf-8") as fp:
        for line in header:
            fp.write(line + "\n")
        for i in range(n_frames):
            t = (float(frames[i]) - float(start_frame)) / float(frame_rate)
            parts = [str(int(frames[i])), f"{t:.6f}"]
            for m in range(n_markers):
                for v in arr[i, m]:
                    if not math.isfinite(float(v)):
                        parts.append("")
                    else:
                        parts.append(f"{float(v):.6f}")
            fp.write("\t".join(parts) + "\n")


def read_trc(path: Path) -> TrcData:
    path = Path(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 5:
        raise ValueError(f"TRC 过短: {path}")

    meta = lines[2].split("\t")
    frame_rate = float(meta[0])
    units = meta[4] if len(meta) > 4 else "m"

    # marker names on line 3: Frame# Time Name1   Name2 ...
    name_parts = lines[3].split("\t")
    marker_names: list[str] = []
    for p in name_parts[2:]:
        p = p.strip()
        if p:
            marker_names.append(p)

    rows_f: list[int] = []
    rows_t: list[float] = []
    rows_xyz: list[list[float]] = []
    for line in lines[5:]:
        if not line.strip():
            continue
        cols = line.split("\t")
        if len(cols) < 2 + 3 * len(marker_names):
            # trailing empties ok
            while len(cols) < 2 + 3 * len(marker_names):
                cols.append("")
        rows_f.append(int(float(cols[0])))
        rows_t.append(float(cols[1]))
        xyz: list[float] = []
        for i in range(len(marker_names)):
            for j in range(3):
                cell = cols[2 + i * 3 + j].strip()
                xyz.append(float(cell) if cell else float("nan"))
        rows_xyz.append(xyz)

    coords = np.asarray(rows_xyz, dtype=np.float64).reshape(
        len(rows_xyz), len(marker_names), 3
    )
    return TrcData(
        marker_names=marker_names,
        frames=np.asarray(rows_f, dtype=np.int64),
        times=np.asarray(rows_t, dtype=np.float64),
        coords_yup=coords,
        frame_rate=frame_rate,
        units=units,
    )
