#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_BIN="${BLENDER_PATH:-/home/leon/Downloads/blender-2.92.0-linux64_custom/blender-2.92.0-linux64/blender}"
LOG_DIR="${EXPORT_OBJ_FRAMES_LOG_DIR:-$SCRIPT_DIR/logs}"
LOG_FILE="${EXPORT_OBJ_FRAMES_LOG:-$LOG_DIR/export_obj_frames_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$LOG_FILE")"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Logging export_obj_frames.sh output to: $LOG_FILE"
echo "Started at: $(date -Is)"

cd "$SCRIPT_DIR"
exec "$BLENDER_BIN" -b -P export_obj_frames.py -- "$@"
