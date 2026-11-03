#!/usr/bin/env bash
# UVC 曝光 preset（开录/预览前调用；Route2 管线本身不设 v4l2）
#
# Preset:
#   auto      光圈优先自动曝光（默认）
#   balanced  手动 ~8ms 快门 + 中等 gain（快动与亮度折中）
#   fast      手动 ~3ms 快门 + 较高 gain（快动优先，需补光）
#
# 用法:
#   bash scripts/route2/set_camera_exposure.sh balanced
#   bash scripts/route2/set_camera_exposure.sh fast --device /dev/video0
#   ROUTE2_CAMERA_EXPOSURE=off bash scripts/route2/run_camera_pose3d.sh  # 跳过
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../../" && pwd)"
DEVICE="${ROUTE2_CAMERA_DEVICE:-/dev/video0}"
PRESET="auto"
DRY_RUN=0

usage() {
  cat <<'EOF'
用法: set_camera_exposure.sh [balanced|fast|auto] [选项]

Preset:
  balanced   手动 exposure=80  gain=45  sharpness=4  (~1/125s)
  fast       手动 exposure=30  gain=55  sharpness=5  (~1/330s，建议补光)
  auto       光圈优先 auto_exposure=3

选项:
  --device PATH   v4l2 设备（默认 /dev/video0 或 ROUTE2_CAMERA_DEVICE）
  --dry-run       只打印将执行的 v4l2-ctl，不写设备
  -h, --help

环境变量（run_camera_pose3d.sh / run_camera_preview.sh）:
  ROUTE2_CAMERA_EXPOSURE=balanced|fast|auto|off   默认 auto；off 跳过
  ROUTE2_CAMERA_DEVICE=/dev/video0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    balanced|fast|auto)
      PRESET="$1"
      shift
      ;;
    --device)
      DEVICE="${2:?--device 需要路径}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "未知参数: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! command -v v4l2-ctl >/dev/null 2>&1; then
  echo "[camera-exposure] 跳过: 未找到 v4l2-ctl" >&2
  exit 0
fi

if [[ ! -e "$DEVICE" ]]; then
  echo "[camera-exposure] 跳过: 设备不存在 $DEVICE" >&2
  exit 0
fi

run_v4l() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[dry-run] v4l2-ctl -d $DEVICE $*"
  else
    v4l2-ctl -d "$DEVICE" "$@"
  fi
}

ctrl_exists() {
  v4l2-ctl -d "$DEVICE" --list-ctrls 2>/dev/null | grep -q "^[[:space:]]*$1 "
}

apply_manual() {
  local exp="$1" gain="$2" sharp="$3"
  if ! ctrl_exists "auto_exposure"; then
    echo "[camera-exposure] 跳过: $DEVICE 无 auto_exposure" >&2
    return 0
  fi
  run_v4l -c auto_exposure=1
  if ctrl_exists "exposure_time_absolute"; then
    run_v4l -c "exposure_time_absolute=${exp}"
  fi
  if ctrl_exists "gain"; then
    run_v4l -c "gain=${gain}"
  fi
  if ctrl_exists "sharpness"; then
    run_v4l -c "sharpness=${sharp}"
  fi
}

case "$PRESET" in
  balanced)
    apply_manual 80 45 4
    LABEL="balanced (~8ms, gain=45)"
    ;;
  fast)
    apply_manual 30 55 5
    LABEL="fast (~3ms, gain=55)"
    ;;
  auto)
    if ctrl_exists "auto_exposure"; then
      run_v4l -c auto_exposure=3
    else
      echo "[camera-exposure] 跳过: $DEVICE 无 auto_exposure" >&2
      exit 0
    fi
    LABEL="auto (aperture priority)"
    ;;
esac

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[camera-exposure] preset=$PRESET device=$DEVICE ($LABEL)"
  exit 0
fi

echo "[camera-exposure] $DEVICE preset=$PRESET ($LABEL)"
if ctrl_exists "auto_exposure"; then
  v4l2-ctl -d "$DEVICE" -C auto_exposure 2>/dev/null || true
fi
if ctrl_exists "exposure_time_absolute"; then
  v4l2-ctl -d "$DEVICE" -C exposure_time_absolute 2>/dev/null || true
fi
if ctrl_exists "gain"; then
  v4l2-ctl -d "$DEVICE" -C gain 2>/dev/null || true
fi
