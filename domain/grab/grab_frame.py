"""一次 grab 的帧句柄（≈ ZED grab() 返回值）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from platform.jetson.prepared_stereo import PreparedStereoFrame


@dataclass
class GrabFrame:
    capture_seq: int
    pull_ms: float
    recording_submitted: bool = False
    prepared: Optional[PreparedStereoFrame] = None
    sbs: Any | None = None
    _released: bool = False

    def consume_prepared(self) -> PreparedStereoFrame:
        """VIC 路径：交给 infer 队列。"""
        if self.prepared is None:
            raise RuntimeError("GrabFrame has no prepared payload")
        prepared = self.prepared
        self.prepared = None
        return prepared

    def consume_sbs(self) -> tuple[Any, float]:
        """GPU 路径：raw SBS + pull_ms 交给 infer。"""
        if self.sbs is None:
            raise RuntimeError("GrabFrame has no sbs payload")
        sbs = self.sbs
        self.sbs = None
        return sbs, self.pull_ms

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if self.prepared is not None:
            try:
                self.prepared.release()
            except Exception:
                pass
            self.prepared = None
        if self.sbs is not None:
            try:
                self.sbs.release()
            except Exception:
                pass
            self.sbs = None
