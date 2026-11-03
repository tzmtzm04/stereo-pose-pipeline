"""Path 6 MSK MJCF 加载（MyoConverter 产出与本地 MuJoCo 版本兼容）。"""

from __future__ import annotations

import re
from pathlib import Path


def ensure_path6_mjcf_compat(model_path: Path | str) -> Path:
    """去掉旧版 MuJoCo 不认识的属性，返回可 load 的 XML 路径。"""
    src = Path(model_path).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(src)

    compat = src.with_name(f"{src.stem}_compat{src.suffix}")
    if compat.is_file() and compat.stat().st_mtime >= src.stat().st_mtime:
        return compat

    text = src.read_text(encoding="utf-8")
    text = re.sub(r'\s*<option collision="predefined"/>\s*', "\n", text, count=1)
    compat.write_text(text, encoding="utf-8")
    return compat
