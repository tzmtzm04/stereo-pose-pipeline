"""整型字段安全读取（getattr 在值为 None 时不会回退 default）。"""

from __future__ import annotations

from typing import Any


def int_attr(obj: Any, name: str, default: int = -1) -> int:
    val = getattr(obj, name, default)
    if val is None:
        return int(default)
    try:
        return int(val)
    except (TypeError, ValueError):
        return int(default)
