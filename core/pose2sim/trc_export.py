"""full_session_pose3d.jsonl → Pose2Sim pose-3d/*.trc（方案 B：跳过 pose/三角化）。"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.pose2sim.coco17_layout import COCO17_MARKER_NAMES, COCO17_P2S_MARKERS

_MIN_CONF = 1e-4


@dataclass(frozen=True)
class TrcExportStats:
    frames_in: int = 0
    frames_written: int = 0
    trc_path: Path | None = None
    trial_dir: Path | None = None


def left_cam_opencv_to_zup(xyz: np.ndarray) -> np.ndarray:
    """左相机 OpenCV (X 右, Y 下, Z 前) → Pose2Sim 三角化后的 Z-up 中间系。"""
    pts = np.asarray(xyz, dtype=np.float32)
    x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
    return np.stack([x, z, -y], axis=-1)


def zup_to_yup(flat_zup: np.ndarray) -> np.ndarray:
    """(N, 3*M) Z-up → Y-up，与 Pose2Sim.common.zup2yup 一致。"""
    out = np.empty_like(flat_zup)
    n_markers = flat_zup.shape[1] // 3
    for i in range(n_markers):
        out[:, i * 3 + 0] = flat_zup[:, i * 3 + 1]
        out[:, i * 3 + 1] = flat_zup[:, i * 3 + 2]
        out[:, i * 3 + 2] = flat_zup[:, i * 3 + 0]
    return out


def _load_pose_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "kpts3d" in row:
                rows.append(row)
    return rows


def _kpts17x4(raw: object) -> np.ndarray | None:
    arr = np.asarray(raw, dtype=np.float32)
    if arr.size == 68:
        return arr.reshape(17, 4)
    if arr.shape == (17, 4):
        return arr
    # anipose optim 写 full 133×4，COCO body 在前 17 行
    if arr.ndim == 2 and arr.shape[0] >= 17 and arr.shape[1] >= 3:
        out = np.full((17, 4), np.nan, dtype=np.float32)
        out[:, :3] = arr[:17, :3]
        if arr.shape[1] > 3:
            out[:, 3] = arr[:17, 3]
        else:
            out[:, 3] = np.where(
                np.all(np.isfinite(arr[:17, :3]), axis=1), 1.0, 0.0
            )
        return out
    return None


def yup_xyz_to_opencv_left(x_yup: float, y_yup: float, z_yup: float) -> np.ndarray:
    """TRC Y-up 单点 → 左相机 OpenCV（方案 B export 互逆）。"""
    x_z = float(z_yup)
    y_z = float(x_yup)
    z_z = float(y_yup)
    return np.array([x_z, -z_z, y_z], dtype=np.float64)


def pose2sim_trc_yup_to_opencv_left(
    x_yup: float, y_yup: float, z_yup: float
) -> np.ndarray:
    """Pose2Sim 原生 tri TRC（Y-up）→ 左相机 OpenCV（= 内部 Z-up / cam_01 系）。"""
    return np.array(
        [float(z_yup), float(x_yup), float(y_yup)], dtype=np.float64
    )


def _frame_rate_from_meta(session_dir: Path, default: float = 60.0) -> float:
    for name in ("full_session_meta.json", "session_meta.json"):
        p = session_dir / name
        if not p.is_file():
            continue
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            fps = float(meta.get("fps") or meta.get("frame_rate") or default)
            if fps > 0:
                return fps
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return default


def _write_trc(
    path: Path,
    *,
    marker_names: tuple[str, ...],
    frames: np.ndarray,
    coords_yup: np.ndarray,
    frame_rate: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = coords_yup.shape[0]
    n_markers = len(marker_names)
    start_frame = int(frames[0])
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
                "m",
                frame_rate,
                start_frame,
                n_frames,
            )
        ),
        "Frame#\tTime\t" + "\t\t\t".join(marker_names) + "\t\t\t",
        "\t\t"
        + "\t".join(f"X{i + 1}\tY{i + 1}\tZ{i + 1}" for i in range(n_markers))
        + "\t",
    ]

    with path.open("w", encoding="utf-8") as fp:
        for line in header:
            fp.write(line + "\n")
        for i in range(n_frames):
            t = (frames[i] - start_frame) / frame_rate
            parts = [str(int(frames[i])), f"{t:.6f}"]
            for m in range(n_markers):
                x, y, z = coords_yup[i, m * 3 : m * 3 + 3]
                for v in (x, y, z):
                    if not math.isfinite(float(v)):
                        parts.append("")
                    else:
                        parts.append(f"{float(v):.6f}")
            fp.write("\t".join(parts) + "\n")


def _ensure_calibration(session_dir: Path, calib_src: Path) -> Path:
    dst = session_dir / "calibration"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("Calib.toml", "Calib.xml"):
        src = calib_src / name
        if src.is_file():
            shutil.copy2(src, dst / name)
    if not (dst / "Calib.toml").is_file():
        raise FileNotFoundError(
            f"找不到 Pose2Sim 标定 Calib.toml（源 {calib_src}）"
        )
    return dst


def _write_config_toml(trial_dir: Path, *, frame_rate: float) -> None:
    text = f"""# 方案 B：外部三角化 TRC → filtering → OpenSim IK
# 由 scripts/offline/export_session_pose2sim_trc.py 生成

[project]
project_dir = '.'
multi_person = false
frame_rate = {frame_rate:g}
frame_range = 'all'

[pose]
pose_model = 'Body'
overwrite_pose = false

[calibration]
calibration_type = 'convert'

[personAssociation.single_person]
tracked_keypoint = 'Nose'

[filtering]
reject_outliers = true
filter = true
type = 'butterworth'
display_figures = false
save_filt_plots = false
make_c3d = false

[filtering.butterworth]
order = 4
cut_off_frequency = 6

[markerAugmentation]
feet_on_floor = false
make_c3d = false

[kinematics]
use_augmentation = false
use_simple_model = true
filter_ik = false
right_left_symmetry = true
default_height = 1.7
remove_individual_scaling_setup = true
remove_individual_ik_setup = true

[logging]
use_custom_logging = true
"""
    (trial_dir / "Config.toml").write_text(text, encoding="utf-8")


def export_rows_to_trc(
    rows: list[dict],
    trc_path: Path,
    *,
    frame_rate: float,
    min_conf: float = _MIN_CONF,
    tri_ok_only: bool = False,
) -> tuple[int, list[int]]:
    """jsonl 行 → Pose2Sim TRC（COCO Body 13 marker）。返回 (写入帧数, capture_seq 列表)。"""
    frames: list[int] = []
    zup_blocks: list[np.ndarray] = []

    for row in rows:
        if tri_ok_only and not row.get("tri_ok", True):
            continue
        kpts = _kpts17x4(row.get("kpts3d"))
        if kpts is None:
            continue
        block = np.full(len(COCO17_MARKER_NAMES) * 3, np.nan, dtype=np.float32)
        ok_any = False
        for mi, (_, coco_i) in enumerate(COCO17_P2S_MARKERS):
            p = kpts[coco_i]
            if float(p[3]) < min_conf or not np.all(np.isfinite(p[:3])):
                continue
            if float(np.linalg.norm(p[:3])) <= 1e-4:
                continue
            block[mi * 3 : mi * 3 + 3] = left_cam_opencv_to_zup(p[:3])
            ok_any = True
        if not ok_any:
            continue
        seq = int(row.get("capture_seq", len(frames)))
        frames.append(seq)
        zup_blocks.append(block)

    if not frames:
        return 0, []

    zup_arr = np.stack(zup_blocks, axis=0)
    yup_flat = zup_to_yup(zup_arr)
    frames_arr = np.asarray(frames, dtype=np.int64)
    _write_trc(
        Path(trc_path),
        marker_names=COCO17_MARKER_NAMES,
        frames=frames_arr,
        coords_yup=yup_flat,
        frame_rate=float(frame_rate),
    )
    return len(frames), frames


def export_session_to_pose2sim_trial(
    pose_path: Path,
    *,
    session_dir: Path,
    trial_name: str = "pose2sim_trial",
    calib_src: Path | None = None,
    frame_rate: float | None = None,
    min_conf: float = _MIN_CONF,
    tri_ok_only: bool = False,
) -> TrcExportStats:
    pose_path = Path(pose_path).expanduser().resolve()
    session_dir = Path(session_dir).expanduser().resolve()
    session_id = session_dir.name

    if calib_src is None:
        calib_src = (
            Path(__file__).resolve().parents[2]
            / "Pose2Sim_Full_Pipeline"
            / "3_pose2sim"
            / "Calibration"
        )
    calib_src = Path(calib_src).expanduser().resolve()
    _ensure_calibration(session_dir, calib_src)

    rows = _load_pose_rows(pose_path)
    stats = TrcExportStats(frames_in=len(rows))
    if not rows:
        return stats

    fps = float(frame_rate or _frame_rate_from_meta(session_dir))
    trial_dir = session_dir / trial_name
    pose3d_dir = trial_dir / "pose-3d"
    trial_dir.mkdir(parents=True, exist_ok=True)
    pose3d_dir.mkdir(parents=True, exist_ok=True)
    _write_config_toml(trial_dir, frame_rate=fps)

    tmp_trc = pose3d_dir / f"{session_id}_export.trc"
    n_written, seqs = export_rows_to_trc(
        rows,
        tmp_trc,
        frame_rate=fps,
        min_conf=min_conf,
        tri_ok_only=tri_ok_only,
    )
    if n_written <= 0:
        return stats
    trc_path = pose3d_dir / f"{session_id}_{seqs[0]}-{seqs[-1]}.trc"
    if tmp_trc != trc_path:
        tmp_trc.rename(trc_path)

    return TrcExportStats(
        frames_in=len(rows),
        frames_written=n_written,
        trc_path=trc_path,
        trial_dir=trial_dir,
    )
