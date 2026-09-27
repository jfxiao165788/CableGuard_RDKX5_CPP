#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
mkdir -p "${BUILD_DIR}"
cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}" -DCABLE_BUILD_ROS2=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "${BUILD_DIR}" --target cable_benchmark -j"$(nproc)"
"${BUILD_DIR}/cable_benchmark" --frames "${CABLE_BENCHMARK_FRAMES:-120}" --warmup 10 \
  --interval "${CABLE_INFERENCE_INTERVAL:-3}" --threads "${CABLE_OPENCV_THREADS:-1}" \
  --json "${ROOT_DIR}/benchmark_cpp.json" --save-image "${ROOT_DIR}/benchmark_cpp.jpg"
python3 "${ROOT_DIR}/python/benchmark_python.py" --frames "${CABLE_BENCHMARK_FRAMES:-120}" \
  --warmup 10 --json "${ROOT_DIR}/benchmark_python.json"
