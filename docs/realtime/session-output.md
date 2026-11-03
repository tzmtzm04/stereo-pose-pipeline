# Session 产物

一次 live 运行对应目录：

```text
outputs/route2_clips/{session_id}/
```

`session_id` 为启动时间戳，例如 `20260706_113020`。

---

## 1. 文件清单

| 文件 | 说明 |
|------|------|
| `session.mjpeg` | 原画：逐帧 JPEG 字节 **顺序拼接**（非标准 MJPEG 容器） |
| `session_index.jsonl` | 每帧在 `session.mjpeg` 中的位置与序号 |
| `session_pose.jsonl` | 推理输出的 3D 关键点，按 `capture_seq` 对齐 |
| `session_meta.json` | 会话元数据与统计 |

契约定义：`contracts/session_bundle.py` → `SessionBundle`。

---

## 2. session_index.jsonl

每行一个 JSON 对象，按写入顺序递增：

```json
{"offset": 0, "size": 377793, "capture_seq": 0, "pts_ns": 0, "is_keyframe": true}
```

| 字段 | 含义 |
|------|------|
| `offset` | 该帧 JPEG 在 `session.mjpeg` 中的起始字节 |
| `size` | JPEG 字节长度 |
| `capture_seq` | 采集序号（全链路时间轴） |
| `pts_ns` | 时间戳（当前常为 0） |
| `is_keyframe` | MJPEG 帧均为 true |

读取第 N 帧：`data[offset : offset + size]`，应以 `FF D8`（JPEG SOI）开头。

---

## 3. session_pose.jsonl

每行一帧推理结果，示例：

```json
{
  "capture_seq": 0,
  "infer_seq": 1,
  "tri_ok": true,
  "kpts3d": [[x, y, z, conf], ...]
}
```

| 字段 | 含义 |
|------|------|
| `capture_seq` | 对应原画序号（可与 index 对齐） |
| `infer_seq` | 推理完成序号（单调递增） |
| `tri_ok` | 三角化是否成功 |
| `kpts3d` | 17 个关节 3D 点 + 置信度 |

**注意：** 推理可能跳过部分 capture 帧（`capture_seq` 非连续），以 infer 实际完成帧为准。录像 index 通常更密。

写入：`domain/l1/pose_session_log.py`，由 infer 路径 append。

---

## 4. session_meta.json

会话结束时写入，常用字段：

```json
{
  "width": 3840,
  "height": 1080,
  "fps": 60,
  "session_id": "20260706_113020",
  "ring_mode": "session_mjpeg",
  "enc_mode": "mjpeg_grab_ingress",
  "layout": "mjpeg_native",
  "calib_path": "/path/to/stereo_sbs.json",
  "duration_sec": 21.4,
  "duration_wall_sec": 22.338,
  "rec_fps": 60.0,
  "frames_submitted": 1284,
  "frames_recorded": 1284,
  "frames_pose": 1180,
  "enc_bytes": 372076922
}
```

| 字段 | 含义 |
|------|------|
| `rec_fps` | 录像帧率（submitted / duration_sec） |
| `frames_recorded` | index 行数 / 写入帧数 |
| `frames_pose` | pose 行数 |
| `enc_bytes` | `session.mjpeg` 字节量 |
| `duration_sec` | 按 capture_seq 跨度 / fps |
| `duration_wall_sec` | 进程墙钟时长 |

标定矩阵 **不复制** 进 session 目录，仅保留 `calib_path` 字符串。

---

## 5. 完整性检查

```bash
# index 行数 ≈ meta.frames_recorded
wc -l outputs/route2_clips/*/session_index.jsonl

# pose 行数 ≈ meta.frames_pose
wc -l outputs/route2_clips/*/session_pose.jsonl

# mjpeg 末帧 offset+size ≤ 文件大小
python3 -c "
import json; from pathlib import Path
d=Path('outputs/route2_clips/20260706_113020')
rows=[json.loads(l) for l in d.joinpath('session_index.jsonl').read_text().splitlines() if l.strip()]
last=rows[-1]; need=last['offset']+last['size']
got=d.joinpath('session.mjpeg').stat().st_size
print('ok' if got>=need else 'truncated', got, need)
"
```

Soak 分析脚本：`scripts/route2/analyze_soak_session.py`。
