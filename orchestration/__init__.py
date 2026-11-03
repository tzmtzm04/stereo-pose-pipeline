"""编排层导出。

Route2LiveSession 懒导出，避免 import orchestration.constants 等轻量模块时
拉起整条 session → pipeline → display 图（也利于 Nuitka 分包编译）。
"""

from __future__ import annotations

from typing import Any

__all__ = ["Route2LiveSession"]


def __getattr__(name: str) -> Any:
    if name == "Route2LiveSession":
        from orchestration.session import Route2LiveSession

        return Route2LiveSession
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")