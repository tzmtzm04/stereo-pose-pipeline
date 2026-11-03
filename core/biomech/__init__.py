"""BioModule v1 离线生物力学推理。"""

from core.biomech.marker_layout import MARKER_COUNT, MARKER_NAMES, MARKER_SKELETON_EDGES
from core.biomech.marker_preview import (
    biomech_contact_summary,
    biomech_row_to_marker_kpts3d,
    load_biomech_by_capture_seq,
)

__all__ = [
    "BINARY_CRITERIA",
    "CRITERIA_DIMS",
    "BioModuleV1",
    "BiomechSessionStats",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_NORM_STATS",
    "DEFAULT_WIN",
    "MARKER_COUNT",
    "MARKER_NAMES",
    "MARKER_SKELETON_EDGES",
    "NormStats",
    "biomech_contact_summary",
    "biomech_row_to_marker_kpts3d",
    "infer_session_poses",
    "load_biomech_by_capture_seq",
    "load_biomodule_v1",
    "load_norm_stats",
    "run_session_biomech",
]


def __getattr__(name: str):
    if name in {
        "BINARY_CRITERIA",
        "CRITERIA_DIMS",
        "BioModuleV1",
    }:
        from core.biomech.model_v1 import BINARY_CRITERIA, CRITERIA_DIMS, BioModuleV1

        return {
            "BINARY_CRITERIA": BINARY_CRITERIA,
            "CRITERIA_DIMS": CRITERIA_DIMS,
            "BioModuleV1": BioModuleV1,
        }[name]
    if name in {
        "DEFAULT_CHECKPOINT",
        "DEFAULT_NORM_STATS",
        "DEFAULT_WIN",
        "NormStats",
        "infer_session_poses",
        "load_biomodule_v1",
        "load_norm_stats",
    }:
        from core.biomech import runtime as _runtime

        return getattr(_runtime, name)
    if name in {"BiomechSessionStats", "run_session_biomech"}:
        from core.biomech import session_biomech as _session

        return getattr(_session, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
