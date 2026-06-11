#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BLENDER_BIN="${BLENDER_PATH:-/home/leon/Downloads/blender-2.92.0-linux64_custom/blender-2.92.0-linux64/blender}"

cd "$SCRIPT_DIR"
exec "$BLENDER_BIN" -b -P export_obj_frames.py -- "$@"
