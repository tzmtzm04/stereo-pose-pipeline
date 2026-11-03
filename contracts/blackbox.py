"""实时黑盒对外契约版本。

交付说明（边界 / API 流程 / 禁止改盒内）：
  docs/realtime/blackbox-delivery.md
"""

from __future__ import annotations

# bump 当 SessionBundle 字段、PoseStore 样本字段或 offline 产出文件名 breaking 变更时
BLACKBOX_CONTRACT_VERSION = 1

# 实时黑盒稳定数据面（不含 GL/imshow/JPEG 等可视化）
REALTIME_DATA_OUTPUTS = (
    "session.mjpeg",
    "session_index.jsonl",
    "session_pose.jsonl",
    "session_meta.json",
)

# 参考可视化（盒外）：CPU live_compose → JPEG；非契约必选
REALTIME_VIZ_REFERENCE = "cpu_jpeg_overlay"
