#!/usr/bin/env bash
set -euo pipefail

WORKSPACE="${CABLE_WORKSPACE_DIR:-/home/sunrise/cable1}"
CPP_DIR="${WORKSPACE}/Cable_Cpp"
source /opt/ros/humble/setup.bash
source "${WORKSPACE}/install/setup.bash" 2>/dev/null || true

# Pin the detector to a selected CPU cluster and leave OpenCV's internal pool
# small; BPU inference is asynchronous and oversubscribing CPU cores hurts ROS.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-1}"
CPU_AFFINITY="${CABLE_CPU_AFFINITY:-4,5,6,7}"
MODEL_PATH="${CABLE_MODEL_PATH:-${WORKSPACE}/models/deployed/current.bin}"

exec taskset -c "${CPU_AFFINITY}" \
  "${CPP_DIR}/build/cable_detection_node" --ros-args \
  -p model_path:="${MODEL_PATH}" \
  -p class_names:="${CABLE_CLASS_NAMES:-defect}" \
  -p conf_thresh:="${CABLE_CONF_THRESH:-0.70}" \
  -p inference_interval:="${CABLE_INFERENCE_INTERVAL:-3}" \
  -p cpu_affinity:="${CPU_AFFINITY}" \
  -p opencv_threads:="${OPENCV_FOR_THREADS_NUM}"
