"""Model_Pose2Sim_simple.osim + COCO17 markers → MyoConverter Step 1 MJCF。"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from core.pose2sim.import_util import _with_stdlib_imports

ROOT = Path(__file__).resolve().parents[2]
OSIM_SETUP = (
    ROOT
    / "Pose2Sim_Full_Pipeline"
    / "pose2sim_src"
    / "Pose2Sim"
    / "OpenSim_Setup"
)
DEFAULT_OSIM = OSIM_SETUP / "Model_Pose2Sim_simple.osim"
DEFAULT_MARKERS = OSIM_SETUP / "Markers_Coco17.xml"
DEFAULT_GEOMETRY = OSIM_SETUP / "Geometry"
DEFAULT_OUT = ROOT / "models" / "mujoco" / "pose2sim_simple"

# path6 三角化 13 marker（Pose2Sim Body；Markers_Coco17 另含 Neck）
PATH6_IK_MARKER_NAMES: tuple[str, ...] = (
    "Nose",
    "RShoulder",
    "RElbow",
    "RWrist",
    "LShoulder",
    "LElbow",
    "LWrist",
    "RHip",
    "RKnee",
    "RAnkle",
    "LHip",
    "LKnee",
    "LAnkle",
)


@dataclass(frozen=True)
class OsimMergeResult:
    osim_out: Path
    marker_count: int


@dataclass(frozen=True)
class MyoConverterRunResult:
    osim_in: Path
    geometry_dir: Path
    output_dir: Path
    mjcf_files: tuple[Path, ...]


def _extract_marker_objects(markers_xml: str) -> str:
    m = re.search(
        r"<MarkerSet\b[^>]*>\s*<objects>(.*?)</objects>",
        markers_xml,
        flags=re.DOTALL,
    )
    if not m:
        raise ValueError("Markers XML 中找不到 MarkerSet/objects")
    inner = m.group(1).strip()
    if not inner:
        raise ValueError("MarkerSet/objects 为空")
    return inner


def merge_coco17_markers_into_osim(
    osim_path: Path,
    markers_path: Path,
    out_path: Path,
) -> OsimMergeResult:
    """将 Markers_Coco17.xml 写入 .osim 的空 MarkerSet。"""
    osim_path = Path(osim_path).expanduser().resolve()
    markers_path = Path(markers_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()

    osim_text = osim_path.read_text(encoding="utf-8")
    markers_text = markers_path.read_text(encoding="utf-8")
    objects_inner = _extract_marker_objects(markers_text)
    n_markers = len(re.findall(r"<Marker\b", objects_inner))

    pattern = (
        r'(<MarkerSet name="markerset">\s*)<objects\s*/>'
        r'(\s*<groups\s*/>)'
    )
    if not re.search(pattern, osim_text):
        raise ValueError(f"{osim_path} 中 MarkerSet/objects 非空或格式不符")
    merged = re.sub(
        pattern,
        rf"\1<objects>\n\t\t\t{objects_inner}\n\t\t</objects>\2",
        osim_text,
        count=1,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(merged, encoding="utf-8")
    return OsimMergeResult(osim_out=out_path, marker_count=n_markers)


def default_merged_osim_path(work_dir: Path | None = None) -> Path:
    base = Path(work_dir or DEFAULT_OUT).expanduser().resolve()
    return base / "Model_Pose2Sim_simple_coco17.osim"


def run_myoconverter_step1_lite(
    osim_path: Path,
    geometry_dir: Path,
    output_dir: Path,
    *,
    add_ground_geom: bool = True,
) -> MyoConverterRunResult:
    """直接调用 xml.converter.convert（无需 OpenSim，适合 ARM64 / Step1 only）。"""
    osim_path = Path(osim_path).expanduser().resolve()
    geometry_dir = Path(geometry_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def _do():
        from myoconverter.xml.converter import convert

        return convert(
            str(osim_path),
            str(output_dir),
            geometry_folder=str(geometry_dir),
            add_ground_geom=bool(add_ground_geom),
            treat_as_normal_path_point=False,
        )

    try:
        mjcf_path = _with_stdlib_imports(_do)
    except ImportError as exc:
        raise ImportError(
            "未安装 MyoConverter lite 依赖。请: bash scripts/offline/setup_myoconverter.sh"
        ) from exc

    mjcf = Path(mjcf_path).expanduser().resolve()
    files = (mjcf,) if mjcf.is_file() else tuple(sorted(output_dir.glob("*.xml")))
    return MyoConverterRunResult(
        osim_in=osim_path,
        geometry_dir=geometry_dir,
        output_dir=output_dir,
        mjcf_files=files,
    )


def run_myoconverter_step1(
    osim_path: Path,
    geometry_dir: Path,
    output_dir: Path,
    *,
    overwrite: bool = True,
    validation: bool = True,
    generate_pdf: bool = True,
) -> MyoConverterRunResult:
    """MyoConverter O2MPipeline Step 1（无肌肉模型，跳过 muscle 优化）。"""
    osim_path = Path(osim_path).expanduser().resolve()
    geometry_dir = Path(geometry_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not osim_path.is_file():
        raise FileNotFoundError(osim_path)
    if not geometry_dir.is_dir():
        raise FileNotFoundError(geometry_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    kwargs = {
        "convert_steps": [1],
        "muscle_list": None,
        "osim_data_overwrite": bool(overwrite),
        "conversion": True,
        "validation": bool(validation),
        "speedy": False,
        "generate_pdf": bool(generate_pdf),
        "add_ground_geom": True,
        "treat_as_normal_path_point": False,
    }

    def _do():
        from myoconverter.O2MPipeline import O2MPipeline

        O2MPipeline(
            str(osim_path),
            str(geometry_dir),
            str(output_dir),
            **kwargs,
        )

    try:
        _with_stdlib_imports(_do)
    except ImportError as exc:
        raise ImportError(
            "未安装 MyoConverter。请先: bash scripts/offline/setup_myoconverter.sh"
        ) from exc

    mjcf = tuple(sorted(output_dir.glob("*_cvt1.xml")))
    if not mjcf:
        mjcf = tuple(sorted(output_dir.glob("*.xml")))
    return MyoConverterRunResult(
        osim_in=osim_path,
        geometry_dir=geometry_dir,
        output_dir=output_dir,
        mjcf_files=mjcf,
    )


def prepare_pose2sim_simple_for_conversion(
    work_dir: Path | None = None,
    *,
    copy_geometry: bool = True,
) -> tuple[Path, Path]:
    """合并 marker 的 .osim + Geometry 目录（可 symlink/copy 到 work_dir）。"""
    work = Path(work_dir or DEFAULT_OUT).expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    merged_osim = default_merged_osim_path(work)
    merge_coco17_markers_into_osim(
        DEFAULT_OSIM, DEFAULT_MARKERS, merged_osim
    )
    geom_dst = work / "Geometry"
    if copy_geometry:
        if geom_dst.exists():
            if geom_dst.is_symlink():
                geom_dst.unlink()
            elif geom_dst.is_dir():
                shutil.rmtree(geom_dst)
        shutil.copytree(DEFAULT_GEOMETRY, geom_dst)
    else:
        if not geom_dst.exists():
            geom_dst.symlink_to(DEFAULT_GEOMETRY.resolve())
    return merged_osim, geom_dst
