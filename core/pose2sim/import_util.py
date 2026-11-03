"""Pose2Sim vendored 包导入（filtering / kinematics 共用）。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_VENDORED = _ROOT / "Pose2Sim_Full_Pipeline" / "pose2sim_src"


def _with_stdlib_imports(fn):
    """repo / cwd 下 platform/ 会遮蔽 stdlib platform，临时移出 sys.path。"""
    repo = str(_ROOT.resolve())
    cwd = str(Path.cwd().resolve())
    removed: list[str] = []
    for p in (repo, cwd, ""):
        while p in sys.path:
            sys.path.remove(p)
            removed.append(p)
    saved_platform = sys.modules.pop("platform", None)
    if _VENDORED.is_dir() and str(_VENDORED) not in sys.path:
        sys.path.insert(0, str(_VENDORED))
    try:
        _load_stdlib_platform()
        return fn()
    finally:
        for p in reversed(removed):
            if p not in sys.path:
                sys.path.insert(0, p)
        if saved_platform is not None and not hasattr(saved_platform, "system"):
            sys.modules["platform"] = saved_platform
        elif "platform" in sys.modules and hasattr(sys.modules["platform"], "system"):
            del sys.modules["platform"]


def _load_stdlib_platform() -> None:
    import importlib.util

    py_ver = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidates = [
        Path(sys.base_prefix) / "lib" / py_ver / "platform.py",
        Path("/usr/lib") / py_ver / "platform.py",
    ]
    for path in candidates:
        if not path.is_file():
            continue
        spec = importlib.util.spec_from_file_location("platform", path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "system"):
            sys.modules["platform"] = mod
            return
    raise RuntimeError("无法加载 stdlib platform（repo platform/ 遮蔽）")


def run_pose2sim_filtering(config_dict: dict) -> None:
    """在 stdlib platform 可用环境下调用 Pose2Sim.filtering。"""

    def _do() -> None:
        from Pose2Sim import Pose2Sim

        Pose2Sim.filtering(config_dict)

    _with_stdlib_imports(_do)


def run_pose2sim_triangulation(config_dict: dict) -> None:
    """在 stdlib platform 可用环境下调用 Pose2Sim.triangulation。"""

    def _do() -> None:
        from Pose2Sim import Pose2Sim

        Pose2Sim.triangulation(config_dict)

    _with_stdlib_imports(_do)


def ensure_pose2sim_import() -> None:
    def _do() -> None:
        import Pose2Sim  # noqa: F401

    try:
        _with_stdlib_imports(_do)
    except ImportError as exc:
        raise RuntimeError(
            "未安装 Pose2Sim。请先运行:\n"
            "  bash scripts/offline/download_pose2sim.sh\n"
            "或: pip install Pose2Sim"
        ) from exc
