"""加载 stereo_sbs.json 标定。"""

from __future__ import annotations

import json
from pathlib import Path

from core.geometry.calib_model import StereoCalibration


def load_stereo_calibration(path: Path) -> StereoCalibration:
    p = Path(path).expanduser().resolve()
    data = json.loads(p.read_text(encoding="utf-8"))
    return StereoCalibration.from_dict(data)
