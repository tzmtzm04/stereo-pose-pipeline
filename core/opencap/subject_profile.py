"""外界输入：受试者身体参数（身高/体重），供 OpenCap LSTM 等使用。

独立于姿态估计管线；可通过 JSON 文件或代码注入。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_MASS_KG = 70.0
DEFAULT_HEIGHT_M = 1.70
SUBJECT_FILENAME = "subject.json"


@dataclass(frozen=True)
class SubjectProfile:
    """OpenCap / OpenSim 所需的人体标量参数。"""

    height_m: float
    mass_kg: float = DEFAULT_MASS_KG
    # 可选元数据
    subject_id: str | None = None
    sex: str | None = None  # "M" | "F" | None
    notes: str | None = None

    def __post_init__(self) -> None:
        h = float(self.height_m)
        m = float(self.mass_kg)
        # 允许外界误传厘米：>3 视为 cm
        if h > 3.0:
            object.__setattr__(self, "height_m", h / 100.0)
        else:
            object.__setattr__(self, "height_m", h)
        object.__setattr__(self, "mass_kg", m)
        if self.height_m <= 0 or self.height_m > 3.0:
            raise ValueError(f"height_m 无效: {self.height_m}（期望约 1.2–2.3 m，或 cm>3）")
        if self.mass_kg <= 0 or self.mass_kg > 400:
            raise ValueError(f"mass_kg 无效: {self.mass_kg}")

    def to_dict(self) -> dict:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None}

    def save(self, path: Path) -> Path:
        path = Path(path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return path


def normalize_height_m(height: float) -> float:
    """米制；>3 按厘米换算。"""
    h = float(height)
    if h > 3.0:
        h = h / 100.0
    return h


def load_subject_profile(path: Path) -> SubjectProfile:
    """从 JSON 加载。支持字段：height_m / height_cm / height，mass_kg / mass / weight_kg。"""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"找不到受试者档案: {path}")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError(f"subject JSON 须为 object: {path}")

    height: float | None = None
    if obj.get("height_m") is not None:
        height = float(obj["height_m"])
    elif obj.get("height_cm") is not None:
        height = float(obj["height_cm"]) / 100.0
    elif obj.get("height") is not None:
        height = normalize_height_m(float(obj["height"]))
    if height is None:
        raise ValueError(f"{path} 缺少 height_m / height_cm / height")

    mass = obj.get("mass_kg", obj.get("mass", obj.get("weight_kg", DEFAULT_MASS_KG)))
    return SubjectProfile(
        height_m=float(height),
        mass_kg=float(mass),
        subject_id=obj.get("subject_id") or obj.get("id"),
        sex=obj.get("sex"),
        notes=obj.get("notes"),
    )


def resolve_subject_profile(
    *,
    subject_path: Path | None = None,
    session_dir: Path | None = None,
    height: float | None = None,
    mass: float | None = None,
    require: bool = False,
) -> SubjectProfile | None:
    """
    解析优先级：
    1. subject JSON（--subject 或 session/opencap/subject.json）
    2. 用 CLI height/mass 覆盖（可只覆盖其一）
    3. 仅 CLI height（mass 默认 70）
    4. None（调用方可再估高）；require=True 则报错
    """
    base: SubjectProfile | None = None

    if subject_path is not None:
        base = load_subject_profile(subject_path)
    elif session_dir is not None:
        session_dir = Path(session_dir)
        for cand in (
            session_dir / "opencap" / SUBJECT_FILENAME,
            session_dir / SUBJECT_FILENAME,
        ):
            if cand.is_file():
                base = load_subject_profile(cand)
                break

    has_h = height is not None and float(height) > 0
    has_m = mass is not None and float(mass) > 0

    if base is not None:
        return SubjectProfile(
            height_m=normalize_height_m(float(height)) if has_h else base.height_m,
            mass_kg=float(mass) if has_m else base.mass_kg,
            subject_id=base.subject_id,
            sex=base.sex,
            notes=base.notes,
        )

    if has_h:
        return SubjectProfile(
            height_m=normalize_height_m(float(height)),
            mass_kg=float(mass) if has_m else DEFAULT_MASS_KG,
        )

    if require:
        raise FileNotFoundError(
            "未提供受试者身体参数。请任选其一：\n"
            "  python3 scripts/offline/set_opencap_subject.py --session ID --height 175 --mass 70\n"
            "  --subject path/to/subject.json\n"
            "  --height 1.75 --mass 70\n"
            f"  或写入 {{session}}/opencap/{SUBJECT_FILENAME}"
        )
    return None


def default_subject_path(session_dir: Path) -> Path:
    return Path(session_dir).expanduser().resolve() / "opencap" / SUBJECT_FILENAME
