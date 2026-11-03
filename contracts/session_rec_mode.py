"""Session 录制 submit 策略（GrabRuntime）。"""

from __future__ import annotations

SESSION_REC_MODE_COPY = "copy"
SESSION_REC_MODE_DEFER = "defer"
SESSION_REC_MODE_POOL = "pool"
SESSION_REC_MODE_REF = "ref"

SESSION_REC_MODES = (
    SESSION_REC_MODE_COPY,
    SESSION_REC_MODE_DEFER,
    SESSION_REC_MODE_POOL,
    SESSION_REC_MODE_REF,
)


def normalize_session_rec_mode(mode: str) -> str:
    key = str(mode).strip().lower()
    if key not in SESSION_REC_MODES:
        raise ValueError(
            f"session_rec_mode must be one of {SESSION_REC_MODES}, got {mode!r}"
        )
    return key
