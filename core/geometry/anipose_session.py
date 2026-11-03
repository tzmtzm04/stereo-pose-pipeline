"""Session 2D 轨迹 → aniposelib triangulate_optim → 3D jsonl。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.geometry.anipose_constraints import (
    constraints_for_joint_set,
    default_joint_indices,
    weak_constraints_for_joint_set,
)
from core.geometry.anipose_filter import (
    AniposeFilter2DConfig,
    AniposeFilter3DConfig,
    filter_pose2d_stereo,
    filter_pose3d_medfilt,
    reproj_error_per_joint,
)
from core.geometry.calib_model import StereoCalibration
from core.geometry.calib_loader import load_stereo_calibration
from core.offline.session_segment import (
    OfflineSessionSegment,
    apply_segment_pose_stack,
)


@dataclass(frozen=True)
class AniposeOptimConfig:
    """默认与 anipose DEFAULT_CONFIG['triangulation'] 一致。"""

    joint_set: str = "body"
    scale_smooth: float = 2.0
    scale_length: float = 2.0
    scale_length_weak: float = 0.5
    reproj_error_threshold: float = 5.0
    reproj_loss: str = "soft_l1"
    n_deriv_smooth: int = 3
    init_ransac: bool = False
    constraints_weak: tuple[tuple[int, int], ...] = ()
    optim_chunk_frames: int = 0  # 0=整段 optim（同官方 optim_chunking 默认关）
    optim_chunk_overlap: int = 15  # 分块时 n_fixed overlap 帧数（官方硬编码 15）


@dataclass
class AniposeOptimStats:
    frames_in_raw: int = 0
    frames_in: int = 0
    frames_out: int = 0
    n_joints: int = 0
    n_cameras: int = 0
    segment: dict | None = None
    optim_chunks: int = 0
    optim_frames: int = 0


@dataclass
class AniposeFilter3DStats:
    frames_in_raw: int = 0
    frames_in: int = 0
    frames_out: int = 0
    n_joints: int = 0
    n_cameras: int = 0
    segment: dict | None = None


def _require_aniposelib():
    try:
        from aniposelib.cameras import Camera, CameraGroup
    except ImportError as exc:
        raise RuntimeError(
            "未安装 aniposelib。请:\n"
            "  pip install 'git+https://github.com/lambdaloop/aniposelib.git'\n"
            "或:\n"
            "  pip install anipose  # 含 aniposelib"
        ) from exc
    return Camera, CameraGroup


def _pad_distortions(dist: np.ndarray, size: int = 14) -> np.ndarray:
    """OpenCV undistortPoints 接受 4/5/8/12/14 参数，勿 pad 到 15。"""
    d = np.zeros(size, dtype=np.float64)
    src = np.asarray(dist, dtype=np.float64).ravel()
    d[: min(len(src), size)] = src[:size]
    return d


def resolve_session_calib_toml(session_dir: Path) -> Path:
    session_dir = session_dir.expanduser().resolve()
    for rel in ("calibration/Calib.toml", "calib/Calib.toml"):
        p = session_dir / rel
        if p.is_file():
            return p.resolve()
    raise FileNotFoundError(f"找不到 session Calib.toml: {session_dir}")


def load_camera_group(calib_path: Path):
    _, CameraGroup = _require_aniposelib()
    return CameraGroup.load(str(Path(calib_path).expanduser().resolve()))


def camera_group_from_stereo(calib: StereoCalibration):
    import cv2

    Camera, CameraGroup = _require_aniposelib()
    w, h = int(calib.image_width), int(calib.image_height)
    rvec1 = np.zeros(3, dtype=np.float64)
    tvec1 = np.zeros(3, dtype=np.float64)
    rvec2, _ = cv2.Rodrigues(np.asarray(calib.R, dtype=np.float64))
    tvec2 = np.asarray(calib.T, dtype=np.float64).reshape(3)
    cam1 = Camera(
        matrix=np.asarray(calib.K1, dtype=np.float64),
        dist=_pad_distortions(calib.D1),
        size=(w, h),
        rvec=rvec1,
        tvec=tvec1,
        name="cam_01",
    )
    cam2 = Camera(
        matrix=np.asarray(calib.K2, dtype=np.float64),
        dist=_pad_distortions(calib.D2),
        size=(w, h),
        rvec=rvec2.ravel(),
        tvec=tvec2,
        name="cam_02",
    )
    return CameraGroup([cam1, cam2], metadata={"source": "stereo_sbs.json"})


def load_camera_group_auto(calib_path: Path):
    p = Path(calib_path).expanduser().resolve()
    if p.suffix.lower() == ".toml":
        return load_camera_group(p)
    if p.suffix.lower() == ".json":
        return camera_group_from_stereo(load_stereo_calibration(p))
    raise ValueError(f"不支持的标定格式: {p}")


def _dedupe_rows_by_capture_seq(rows: list[dict]) -> list[dict]:
    by_seq: dict[int, dict] = {}
    for row in rows:
        by_seq[int(row.get("capture_seq", -1))] = row
    return [by_seq[k] for k in sorted(by_seq)]


def _parse_kpts2d(raw) -> np.ndarray | None:
    if raw is None:
        return None
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return None
    return arr


def load_pose2d_from_jsonl(
    path: Path,
    *,
    joint_indices: tuple[int, ...],
    dedupe_capture_seq: bool = True,
    max_frames: int = 0,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """读 pose jsonl，堆叠为 C×N×J×2 与 scores（左=cam0，右=cam1）。"""
    raw_rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw_rows.append(json.loads(line))
    rows = _dedupe_rows_by_capture_seq(raw_rows) if dedupe_capture_seq else raw_rows
    if max_frames > 0:
        rows = rows[:max_frames]

    n_frames = len(rows)
    n_joints = len(joint_indices)
    points = np.full((2, n_frames, n_joints, 2), np.nan, dtype=np.float64)
    scores = np.zeros((2, n_frames, n_joints), dtype=np.float64)

    for fi, row in enumerate(rows):
        for ci, key in enumerate(("det_left_kpts", "det_right_kpts")):
            kpts = _parse_kpts2d(row.get(key))
            if kpts is None:
                continue
            for ji, src_j in enumerate(joint_indices):
                if src_j < 0 or src_j >= kpts.shape[0]:
                    continue
                conf = float(kpts[src_j, 2])
                scores[ci, fi, ji] = conf
                if np.all(np.isfinite(kpts[src_j, :2])):
                    points[ci, fi, ji, 0] = kpts[src_j, 0]
                    points[ci, fi, ji, 1] = kpts[src_j, 1]

    return points, scores, rows


def load_pose3d_stack_from_rows(
    rows: list[dict],
    *,
    joint_indices: tuple[int, ...],
) -> np.ndarray:
    """N×J×3，从 jsonl kpts3d 抽取 optim 关节子集。"""
    n_frames = len(rows)
    n_joints = len(joint_indices)
    p3ds = np.full((n_frames, n_joints, 3), np.nan, dtype=np.float64)
    for fi, row in enumerate(rows):
        raw = row.get("kpts3d")
        if raw is None:
            continue
        kpts3d = np.asarray(raw, dtype=np.float64)
        if kpts3d.ndim != 2 or kpts3d.shape[1] < 3:
            continue
        for ji, src_j in enumerate(joint_indices):
            if src_j < 0 or src_j >= kpts3d.shape[0]:
                continue
            xyz = kpts3d[src_j, :3]
            w = float(kpts3d[src_j, 3]) if kpts3d.shape[1] > 3 else 1.0
            if w > 0 and np.all(np.isfinite(xyz)):
                p3ds[fi, ji] = xyz
    return p3ds


def _both_cams_observable(points: np.ndarray) -> np.ndarray:
    """N×J：左右目 2D 均有限才参与三角化。"""
    pts = np.asarray(points, dtype=np.float64)
    per_cam = np.all(np.isfinite(pts), axis=-1)
    return np.all(per_cam, axis=0)


def _mask_points_for_optim(points: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.array(points, dtype=np.float64, copy=True)
    out[:, ~valid] = np.nan
    return out


def _mask_p3ds_output(p3ds: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.array(p3ds, dtype=np.float64, copy=True)
    out[~valid] = np.nan
    return out


def _sanitize_optim_x0(x0: np.ndarray, *, n_p3d: int) -> np.ndarray:
    out = np.asarray(x0, dtype=np.float64).copy()
    p3d_flat = out[:n_p3d]
    p3d_flat[~np.isfinite(p3d_flat)] = 0.0
    out[:n_p3d] = p3d_flat
    if out.size > n_p3d:
        lengths = out[n_p3d:]
        good = lengths[np.isfinite(lengths) & (lengths > 0)]
        med = float(np.median(good)) if good.size else 1e-3
        if not np.isfinite(med) or med <= 0:
            med = 1e-3
        bad = (~np.isfinite(lengths)) | (lengths <= 0)
        lengths[bad] = med
        out[n_p3d:] = lengths
    return out


def _optim_points_safe_scale(
    camera_group,
    points: np.ndarray,
    p3ds: np.ndarray,
    *,
    scores: np.ndarray | None,
    constraints: list[list[int]],
    constraints_weak: list[list[int]],
    cfg: AniposeOptimConfig,
    n_fixed: int = 0,
) -> np.ndarray:
    """aniposelib optim_points；修 default_smooth 除零；支持 n_fixed overlap。"""
    from aniposelib.cameras import interpolate_data, medfilt_data
    from scipy import optimize

    constraints_arr = np.asarray(constraints)
    constraints_weak_arr = np.asarray(constraints_weak)
    p3ds_intp = np.apply_along_axis(interpolate_data, 0, p3ds)
    n_fixed = max(0, int(n_fixed))
    if n_fixed > 0:
        p3ds_fixed = p3ds_intp[:n_fixed]
    else:
        p3ds_fixed = None
    p3ds_med = np.apply_along_axis(medfilt_data, 0, p3ds_intp, size=7)
    diff = np.abs(np.diff(p3ds_med, axis=0))
    mean_diff = float(np.nanmean(diff))
    if not np.isfinite(mean_diff) or mean_diff <= 0:
        default_smooth = 1.0
    else:
        default_smooth = 1.0 / mean_diff
    scale_smooth_full = float(cfg.scale_smooth) * default_smooth

    x0 = camera_group._initialize_params_triangulation(
        p3ds_intp, constraints_arr, constraints_weak_arr
    )
    x0 = _sanitize_optim_x0(x0, n_p3d=int(p3ds.size))

    jac = camera_group._jac_sparsity_triangulation(
        points,
        constraints_arr,
        constraints_weak_arr,
        int(cfg.n_deriv_smooth),
    )
    opt2 = optimize.least_squares(
        camera_group._error_fun_triangulation,
        x0=x0,
        jac_sparsity=jac,
        loss="linear",
        ftol=1e-3,
        args=(
            points,
            constraints_arr,
            constraints_weak_arr,
            scores,
            scale_smooth_full,
            float(cfg.scale_length),
            float(cfg.scale_length_weak),
            float(cfg.reproj_error_threshold),
            str(cfg.reproj_loss),
            int(cfg.n_deriv_smooth),
            p3ds_fixed,
        ),
    )
    p3ds_new = opt2.x[: p3ds.size].reshape(p3ds.shape)
    if n_fixed > 0:
        p3ds_new = np.vstack([p3ds_fixed, p3ds_new[n_fixed:]])
    return p3ds_new


def _dlt_init_all(
    camera_group,
    points: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """全序列 DLT 初始化 N×J×3（与官方 triangulate 一致）。"""
    n_cams, n_frames, n_joints, _ = points.shape
    pts_m = _mask_points_for_optim(points, valid)
    flat = camera_group.triangulate(pts_m.reshape(n_cams, n_frames * n_joints, 2))
    p3ds = np.asarray(flat, dtype=np.float64).reshape(n_frames, n_joints, 3)
    return _mask_p3ds_output(p3ds, valid)


def _run_optim_chunked(
    camera_group,
    points: np.ndarray,
    valid: np.ndarray,
    *,
    scores: np.ndarray | None,
    constraints: list[list[int]],
    constraints_weak: list[list[int]],
    cfg: AniposeOptimConfig,
) -> tuple[np.ndarray, int, int]:
    """官方 optim_chunking：全序列 DLT init + 分块 optim + n_fixed overlap 拼接。"""
    n_frames = int(points.shape[1])
    chunk_size = max(1, int(cfg.optim_chunk_frames))
    n_overlap = max(0, int(cfg.optim_chunk_overlap))

    pts_m = _mask_points_for_optim(points, valid)
    p3ds_init = _dlt_init_all(camera_group, points, valid)

    if int(np.sum(np.isfinite(p3ds_init[:, :, 0]))) < 20:
        raise RuntimeError("triangulate_optim: 有效 3D 点不足，无法分块 optim")

    out = np.full((n_frames, points.shape[2], 3), np.nan, dtype=np.float64)
    fixed: np.ndarray | None = None
    n_chunks = 0
    optim_frames = 0

    for start in range(0, n_frames, chunk_size):
        end = min(start + chunk_size, n_frames)
        if start == 0:
            n_fixed = 0
            win_start = start
        else:
            n_fixed = n_overlap
            win_start = start - n_fixed

        pts_win = pts_m[:, win_start:end]
        if start == 0 or fixed is None:
            p3_win = p3ds_init[win_start:end]
        else:
            p3_win = np.vstack([fixed, p3ds_init[start:end]])

        sc = scores[:, win_start:end] if scores is not None else None
        tag = f"{start}-{end - 1}"
        try:
            part = _optim_points_safe_scale(
                camera_group,
                pts_win,
                p3_win,
                scores=sc,
                constraints=constraints,
                constraints_weak=constraints_weak,
                cfg=cfg,
                n_fixed=n_fixed,
            )
        except (RuntimeError, ValueError) as exc:
            print(
                f"WARN: triangulate_optim 跳过 chunk [{tag}]: {exc}",
                flush=True,
            )
            fallback = _mask_p3ds_output(p3ds_init[start:end], valid[start:end])
            out[start:end] = fallback
            if n_overlap > 0 and fallback.shape[0] >= n_overlap:
                fixed = fallback[-n_overlap:]
            continue

        new_part = _mask_p3ds_output(part[n_fixed:], valid[start:end])
        out[start:end] = new_part
        n_chunks += 1
        optim_frames += end - start
        if n_overlap > 0:
            fixed = part[-n_overlap:]

    if n_chunks == 0:
        raise RuntimeError("triangulate_optim 全部分块失败")
    return out, n_chunks, optim_frames


def run_anipose_triangulate_optim(
    points: np.ndarray,
    *,
    camera_group,
    cfg: AniposeOptimConfig,
    scores: np.ndarray | None = None,
) -> tuple[np.ndarray, int, int]:
    """C×N×J×2 → N×J×3。返回 (p3ds, optim_chunks, optim_frames)。"""
    constraints = constraints_for_joint_set(cfg.joint_set)
    weak_pairs = (
        cfg.constraints_weak
        if cfg.constraints_weak
        else weak_constraints_for_joint_set(cfg.joint_set)
    )
    weak = [[int(a), int(b)] for a, b in weak_pairs]
    points_in = np.asarray(points, dtype=np.float64)
    n_frames = int(points_in.shape[1])
    chunk_n = int(cfg.optim_chunk_frames)
    # 与官方一致：短序列或 chunk+100 覆盖全长时不分块
    if chunk_n <= 0 or chunk_n + 100 > n_frames:
        p3ds = camera_group.triangulate_optim(
            points_in,
            init_ransac=bool(cfg.init_ransac),
            constraints=constraints,
            constraints_weak=weak,
            scale_smooth=float(cfg.scale_smooth),
            scale_length=float(cfg.scale_length),
            scale_length_weak=float(cfg.scale_length_weak),
            reproj_error_threshold=float(cfg.reproj_error_threshold),
            reproj_loss=str(cfg.reproj_loss),
            n_deriv_smooth=int(cfg.n_deriv_smooth),
        )
        return np.asarray(p3ds, dtype=np.float64), 1, n_frames

    valid = _both_cams_observable(points_in)
    p3ds, n_chunks, optim_frames = _run_optim_chunked(
        camera_group,
        points_in,
        valid,
        scores=scores,
        constraints=constraints,
        constraints_weak=weak,
        cfg=cfg,
    )
    return _mask_p3ds_output(p3ds, valid), n_chunks, optim_frames


def full_kpts_for_joint_indices(joint_indices: tuple[int, ...]) -> int:
    """jsonl kpts3d 行宽：COCO body=17，body_feet=23。"""
    if not joint_indices:
        return 17
    return int(max(joint_indices)) + 1


def write_anipose_pose3d_jsonl(
    *,
    template_rows: list[dict],
    p3ds: np.ndarray,
    joint_indices: tuple[int, ...],
    out_path: Path,
    full_kpts: int | None = None,
    anipose_filter_2d: bool = False,
    anipose_filter_3d: bool = False,
) -> None:
    """将 optim 结果写回 jsonl；仅更新 joint_indices 对应点，其余填 NaN。"""
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_frames, n_joints = p3ds.shape[0], p3ds.shape[1]
    if n_frames != len(template_rows):
        raise ValueError(
            f"帧数不一致: p3ds {n_frames} vs rows {len(template_rows)}"
        )
    kpts_width = int(full_kpts if full_kpts is not None else full_kpts_for_joint_indices(joint_indices))

    with out_path.open("w", encoding="utf-8") as out_f:
        for fi, row in enumerate(template_rows):
            kpts3d = np.full((kpts_width, 4), np.nan, dtype=np.float64)
            kpts3d[:, 3] = 0.0
            for ji, src_j in enumerate(joint_indices):
                xyz = p3ds[fi, ji]
                if np.all(np.isfinite(xyz)):
                    kpts3d[src_j, :3] = xyz
                    kpts3d[src_j, 3] = 1.0
            rec = dict(row)
            rec["kpts3d"] = kpts3d.tolist()
            rec["tri_ok"] = bool(np.any(np.isfinite(kpts3d[:, :3])))
            rec["anipose_optim"] = True
            rec["anipose_filter2d"] = bool(anipose_filter_2d)
            rec["anipose_filter3d"] = bool(anipose_filter_3d)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_session_anipose_optim(
    *,
    pose_path: Path,
    calib_path: Path,
    out_path: Path,
    cfg: AniposeOptimConfig | None = None,
    filter_2d: AniposeFilter2DConfig | None = None,
    filter_3d: AniposeFilter3DConfig | None = None,
    segment: OfflineSessionSegment | None = None,
    max_frames: int = 0,
) -> AniposeOptimStats:
    cfg = cfg or AniposeOptimConfig()
    f2d = filter_2d if filter_2d is not None else AniposeFilter2DConfig()
    f3d = filter_3d if filter_3d is not None else AniposeFilter3DConfig()
    joint_indices = default_joint_indices(cfg.joint_set)

    raw_count = sum(1 for _ in Path(pose_path).open(encoding="utf-8"))
    points, scores, rows = load_pose2d_from_jsonl(
        pose_path,
        joint_indices=joint_indices,
        max_frames=int(max_frames),
    )
    points, scores, rows, seg_used = apply_segment_pose_stack(
        points, scores, rows, segment
    )
    cgroup = load_camera_group_auto(calib_path)

    if f2d.enabled:
        points, scores = filter_pose2d_stereo(points, scores, f2d)

    p3ds, n_chunks, optim_frames = run_anipose_triangulate_optim(
        points,
        camera_group=cgroup,
        cfg=cfg,
        scores=scores,
    )

    if f3d.enabled:
        reproj_err = reproj_error_per_joint(cgroup, points, p3ds)
        p3ds = filter_pose3d_medfilt(
            p3ds, reproj_err, f3d, joint_indices=joint_indices
        )

    write_anipose_pose3d_jsonl(
        template_rows=rows,
        p3ds=p3ds,
        joint_indices=joint_indices,
        out_path=out_path,
        anipose_filter_2d=f2d.enabled,
        anipose_filter_3d=f3d.enabled,
    )

    stats = AniposeOptimStats(
        frames_in_raw=raw_count,
        frames_in=len(rows),
        frames_out=len(rows),
        n_joints=len(joint_indices),
        n_cameras=len(cgroup.cameras),
        segment=seg_used.to_dict() if seg_used is not None else None,
        optim_chunks=n_chunks,
        optim_frames=optim_frames,
    )
    return stats


def run_session_filter3d(
    *,
    pose3d_path: Path,
    calib_path: Path,
    out_path: Path,
    cfg: AniposeFilter3DConfig | None = None,
    joint_set: str = "body",
    segment: OfflineSessionSegment | None = None,
    max_frames: int = 0,
) -> AniposeFilter3DStats:
    """Stage 3：tri_raw_3d → 重投影门控 + 3D medfilt → tri_filtered_3d。"""
    f3d = cfg if cfg is not None else AniposeFilter3DConfig()
    if not f3d.enabled:
        raise ValueError("filter3d.enabled 为 False，无输出")

    joint_indices = default_joint_indices(joint_set)
    raw_count = sum(1 for _ in Path(pose3d_path).open(encoding="utf-8"))

    points, _scores, rows = load_pose2d_from_jsonl(
        pose3d_path,
        joint_indices=joint_indices,
        max_frames=int(max_frames),
    )
    points, _scores, rows, seg_used = apply_segment_pose_stack(
        points, _scores, rows, segment
    )
    p3ds = load_pose3d_stack_from_rows(rows, joint_indices=joint_indices)
    cgroup = load_camera_group_auto(calib_path)

    reproj_err = reproj_error_per_joint(cgroup, points, p3ds)
    p3ds_f = filter_pose3d_medfilt(
        p3ds, reproj_err, f3d, joint_indices=joint_indices
    )

    write_anipose_pose3d_jsonl(
        template_rows=rows,
        p3ds=p3ds_f,
        joint_indices=joint_indices,
        out_path=out_path,
        anipose_filter_2d=False,
        anipose_filter_3d=True,
    )

    return AniposeFilter3DStats(
        frames_in_raw=raw_count,
        frames_in=len(rows),
        frames_out=len(rows),
        n_joints=len(joint_indices),
        n_cameras=len(cgroup.cameras),
        segment=seg_used.to_dict() if seg_used is not None else None,
    )
