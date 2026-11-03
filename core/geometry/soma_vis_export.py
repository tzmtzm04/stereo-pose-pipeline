"""session SOMA jsonl → 官方 demo_soma_vis 可用的 motion / npz。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from core.geometry.rotations import axis_angle_from_matrix, rodrigues
from core.geometry.soma_skeleton import SomaSkeleton, default_soma_skeleton

# OpenCV 左目 ↔ SOMA Y-up（与 export_soma_skeleton.py 一致，自逆）
_OPENCV_TO_SOMA = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)


@dataclass
class SomaMotionExportStats:
    frames_in: int = 0
    frames_out: int = 0
    fps: float = 30.0


def opencv_to_soma_pose_trans(
    pose77: np.ndarray,
    trans: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """(77,3) 轴角 + (3,) 平移：OpenCV → SOMA Y-up 米制。"""
    pose = np.asarray(pose77, dtype=np.float64).reshape(77, 3)
    t = np.asarray(trans, dtype=np.float64).reshape(3)
    t_soma = (_OPENCV_TO_SOMA @ t).astype(np.float32)
    pose_soma = np.zeros((77, 3), dtype=np.float32)
    for i in range(77):
        R = rodrigues(pose[i])
        R_s = _OPENCV_TO_SOMA @ R @ _OPENCV_TO_SOMA
        pose_soma[i] = axis_angle_from_matrix(R_s)
    return pose_soma, t_soma


def _rest_local_soma(skeleton: SomaSkeleton) -> np.ndarray:
    """rest_local 从 OpenCV JSON 映回 SOMA 局部骨向量。"""
    rest = skeleton.rest_local.astype(np.float64).copy()
    for i in range(rest.shape[0]):
        rest[i] = _OPENCV_TO_SOMA @ rest[i]
    return rest


def pose77_trans_to_demo_se3(
    pose77: np.ndarray,
    trans: np.ndarray,
    skeleton: SomaSkeleton,
    *,
    in_opencv: bool = True,
) -> np.ndarray:
    """单帧 → demo_soma_vis motion 切片 (78, 4, 4)。

    demo 读取：
      joint_rot = motion[..., :3, :3]
      root_trans = motion[..., 1, :3, 3]  # Hips
    """
    if in_opencv:
        pose77, trans = opencv_to_soma_pose_trans(pose77, trans)
    pose = np.asarray(pose77, dtype=np.float64).reshape(77, 3)
    trans = np.asarray(trans, dtype=np.float64).reshape(3)
    rest = _rest_local_soma(skeleton)

    se3 = np.zeros((78, 4, 4), dtype=np.float32)
    se3[:, 3, 3] = 1.0
    se3[0, :3, :3] = np.eye(3, dtype=np.float32)
    for j in range(1, 78):
        se3[j, :3, :3] = rodrigues(pose[j - 1]).astype(np.float32)
        se3[j, :3, 3] = rest[j].astype(np.float32)
    se3[1, :3, 3] = trans.astype(np.float32)
    return se3


def load_soma_pose_sequence_from_jsonl(
    path: Path,
    *,
    start_seq: int | None = None,
    end_seq: int | None = None,
    stride: int = 1,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """返回 [(capture_seq, pose77, trans), ...]。"""
    path = Path(path).expanduser().resolve()
    out: list[tuple[int, np.ndarray, np.ndarray]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seq = int(rec.get("capture_seq", -1))
            if start_seq is not None and seq < int(start_seq):
                continue
            if end_seq is not None and seq > int(end_seq):
                continue
            if "soma_pose77" not in rec or "soma_trans" not in rec:
                continue
            pose = np.asarray(rec["soma_pose77"], dtype=np.float32).reshape(77, 3)
            trans = np.asarray(rec["soma_trans"], dtype=np.float32).reshape(3)
            out.append((seq, pose, trans))
    if stride > 1:
        out = out[:: int(stride)]
    return out


def export_demo_motion_npy(
    frames: list[tuple[int, np.ndarray, np.ndarray]],
    out_path: Path,
    *,
    skeleton: SomaSkeleton | None = None,
    fps: float = 30.0,
) -> SomaMotionExportStats:
    """导出 (T, 78, 4, 4) float32 .npy，供 tools/demo_soma_vis.py --motion-file。"""
    skeleton = skeleton or default_soma_skeleton()
    out_path = Path(out_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise ValueError("无 soma_pose77/soma_trans 帧可导出")

    motion = np.stack(
        [
            pose77_trans_to_demo_se3(pose, trans, skeleton, in_opencv=True)
            for _, pose, trans in frames
        ],
        axis=0,
    ).astype(np.float32)
    np.save(out_path, motion)

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(
        json.dumps(
            {
                "format": "demo_soma_vis_motion",
                "shape": list(motion.shape),
                "fps": float(fps),
                "coord_frame_out": "soma_y_up_meters",
                "coord_frame_in": "opencv_left_y_down",
                "capture_seqs": [int(s) for s, _, _ in frames],
                "tool": "tools/demo_soma_vis.py --motion-file",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return SomaMotionExportStats(
        frames_in=len(frames),
        frames_out=int(motion.shape[0]),
        fps=float(fps),
    )


def export_soma_npz_optional(
    frames: list[tuple[int, np.ndarray, np.ndarray]],
    out_path: Path,
    *,
    identity_model_type: str = "mhr",
    identity_coeffs_dim: int = 45,
) -> None:
    """可选：官方 soma.io.save_soma_npz（需 py-soma-x + torch）。"""
    try:
        from soma.io import save_soma_npz
    except ImportError as exc:
        raise RuntimeError("导出 npz 需要 py-soma-x: pip install py-soma-x") from exc

    skeleton = default_soma_skeleton()
    poses = []
    trans = []
    for _, pose, t in frames:
        p, tr = opencv_to_soma_pose_trans(pose, t)
        poses.append(p)
        trans.append(tr)
    poses_np = np.stack(poses, axis=0).astype(np.float32)
    transl_np = np.stack(trans, axis=0).astype(np.float32)
    joint_names = [str(n) for n in skeleton.joint_names[1:]]

    save_soma_npz(
        out_path,
        poses_np,
        transl_np,
        joint_names=joint_names,
        identity_model_type=str(identity_model_type),
        identity_coeffs=np.zeros((1, int(identity_coeffs_dim)), dtype=np.float32),
        unit="meters",
        keep_root=False,
    )
