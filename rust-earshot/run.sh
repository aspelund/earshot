#!/bin/bash
export ORT_DYLIB_PATH="$(dirname "$0")/onnxruntime-linux-x64-1.22.0/lib/libonnxruntime.so"
exec "$(dirname "$0")/target/release/earshot" "$@"
