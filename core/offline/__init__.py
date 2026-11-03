"""离线层：session 重算 3D pose。

子模块按需 import；勿在此 eager import pipeline/jetson，避免 preview 脚本触发
`platform` 包与 stdlib 冲突。
"""

from __future__ import annotations

__all__: list[str] = []
