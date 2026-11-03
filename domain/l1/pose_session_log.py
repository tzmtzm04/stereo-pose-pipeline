"""长会话 3D pose 追加写盘（供后处理按 capture_seq 切 clip）。"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from domain.l1.pose_history import Pose3DSample


class L1PoseSessionLog:
    """线程安全 append-only JSONL。"""

    def __init__(self, path: Path) -> None:
        self._path = Path(path).expanduser().resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._fp = self._path.open("a", encoding="utf-8")
        self._count = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def count(self) -> int:
        return self._count

    def append_sample(self, sample: Pose3DSample) -> None:
        row: dict = {
            "capture_seq": sample.capture_seq,
            "infer_seq": sample.infer_seq,
            "tri_ok": sample.tri_ok,
        }
        if sample.kpts3d is not None:
            row["kpts3d"] = sample.kpts3d.tolist()
        if sample.opencap20_kpts3d is not None:
            row["opencap20_kpts3d"] = sample.opencap20_kpts3d.tolist()
        if sample.det_left is not None:
            kp = sample.det_left.get("keypoints")
            if kp is not None:
                row["det_left_kpts"] = np.asarray(kp, dtype=np.float32).tolist()
            kp_hypo = sample.det_left.get("keypoints_hypo")
            if kp_hypo is not None:
                row["det_left_kpts_hypo"] = np.asarray(kp_hypo, dtype=np.float32).tolist()
            box = sample.det_left.get("box_xyxy")
            if box is not None:
                row["det_left_box"] = [int(v) for v in box[:4]]
            det_conf = sample.det_left.get("conf")
            if det_conf is not None:
                row["det_left_box_conf"] = float(det_conf)
        if sample.det_right is not None:
            kp = sample.det_right.get("keypoints")
            if kp is not None:
                row["det_right_kpts"] = np.asarray(kp, dtype=np.float32).tolist()
            kp_hypo = sample.det_right.get("keypoints_hypo")
            if kp_hypo is not None:
                row["det_right_kpts_hypo"] = np.asarray(
                    kp_hypo, dtype=np.float32
                ).tolist()
            box = sample.det_right.get("box_xyxy")
            if box is not None:
                row["det_right_box"] = [int(v) for v in box[:4]]
            det_conf = sample.det_right.get("conf")
            if det_conf is not None:
                row["det_right_box_conf"] = float(det_conf)
        if sample.soma_joints78 is not None:
            row["soma_joints78"] = np.asarray(
                sample.soma_joints78, dtype=np.float32
            ).tolist()
        if sample.body_loss is not None:
            row["body_loss"] = float(sample.body_loss)
        if sample.filter_2d is not None:
            row["filter_2d"] = str(sample.filter_2d)
        line = json.dumps(row, ensure_ascii=False)
        with self._lock:
            self._fp.write(line + "\n")
            self._count += 1

    def flush(self) -> None:
        with self._lock:
            self._fp.flush()

    def close(self) -> None:
        with self._lock:
            self._fp.close()

    @staticmethod
    def export_range(
        path: Path,
        start_capture_seq: int,
        end_capture_seq: int,
    ) -> list[dict]:
        start, end = int(start_capture_seq), int(end_capture_seq)
        rows: list[dict] = []
        if not Path(path).is_file():
            return rows
        with Path(path).open(encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                seq = int(row.get("capture_seq", -1))
                if start <= seq <= end:
                    rows.append(row)
        return rows


PoseSidecar = L1PoseSessionLog
