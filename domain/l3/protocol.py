"""L3 动作判段插件接口。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from domain.l1.pose_history import Pose3DSample


@dataclass(frozen=True)
class L3FrameContext:
    capture_seq: int
    latest: Optional[Pose3DSample]
    history_len: int


class ActionDetector(Protocol):
    """L3 判段插件：infer 路径每帧调用 on_frame；poll 由 runtime 调度。"""

    def on_capture_seq(self, capture_seq: int) -> None: ...

    def stop(self) -> None: ...
