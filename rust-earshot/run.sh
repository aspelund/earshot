#!/bin/bash
# Use ONNX runtime from Python venv
export ORT_DYLIB_PATH="/home/mattias/projects/earshot/.venv/lib/python3.12/site-packages/onnxruntime/capi/libonnxruntime.so.1.23.0"

# Force X11 on WSL2 (Wayland has issues with egui)
export WAYLAND_DISPLAY=

# Use Glow (OpenGL) renderer on WSL2 - wgpu has GPU access issues
if grep -qi microsoft /proc/version 2>/dev/null; then
    export EARSHOT_USE_GLOW=1
fi

exec "$(dirname "$0")/target/release/earshot" "$@"
