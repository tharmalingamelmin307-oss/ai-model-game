#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${RKNN_RUNTIME_DIR:-/home/orangepi/Desktop/rknn_runtime}"

cd "$ROOT_DIR"

COMMON_FLAGS=(
  -O3 -DNDEBUG -std=c++17 -mcpu=cortex-a76
  -I.
  -I"$RUNTIME_DIR/include"
)

COMMON_SOURCES=(
  postprocess/full_io_runtime.cpp
  postprocess/nmsfree_topk_selector.cpp
  postprocess/roi_mask_decoder.cpp
  postprocess/seg_class_selector.cpp
)

COMMON_LIBS=(
  -L"$RUNTIME_DIR/lib"
  -Wl,-rpath="$RUNTIME_DIR/lib"
  -lrknnrt -pthread
)

g++ "${COMMON_FLAGS[@]}" \
  $(pkg-config --cflags opencv4) \
  seg_worker.cpp "${COMMON_SOURCES[@]}" \
  -o seg_worker \
  $(pkg-config --libs opencv4) \
  "${COMMON_LIBS[@]}"

g++ "${COMMON_FLAGS[@]}" \
  $(pkg-config --cflags opencv4) \
  det_worker.cpp "${COMMON_SOURCES[@]}" \
  -o det_worker \
  $(pkg-config --libs opencv4) \
  "${COMMON_LIBS[@]}"

echo "built: $ROOT_DIR/seg_worker"
echo "built: $ROOT_DIR/det_worker"
