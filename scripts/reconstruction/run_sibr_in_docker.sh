#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_NAME="${1:-sibr-builder:ubuntu22.04-cuda11.8}"
MODEL_PATH="${2:-${ROOT_DIR}/outputs/3dgs_tandt_30000iter}"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker command not found."
  echo "Install Docker first, then retry."
  exit 1
fi

DOCKER_CMD="docker"
if ! docker info >/dev/null 2>&1; then
  if command -v sudo >/dev/null 2>&1; then
    DOCKER_CMD="sudo docker"
  else
    echo "Error: cannot access Docker daemon and sudo is unavailable."
    exit 1
  fi
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "Error: model path does not exist: ${MODEL_PATH}"
  exit 1
fi

if [[ ! -f "${MODEL_PATH}/cfg_args" ]]; then
  echo "Error: cfg_args not found under model path: ${MODEL_PATH}"
  echo "Tip: pass your 3DGS output folder as arg2, e.g. outputs/3dgs_tandt_30000iter"
  exit 1
fi

CONTAINER_MODEL_PATH="${MODEL_PATH}"
if [[ "${MODEL_PATH}" == "${ROOT_DIR}"* ]]; then
  REL_PATH="${MODEL_PATH#${ROOT_DIR}/}"
  CONTAINER_MODEL_PATH="/workspace/${REL_PATH}"
fi

APP_PATH="/workspace/third_party/gaussian-splatting/SIBR_viewers/install/bin/SIBR_gaussianViewer_app"
APP_ROOT="/workspace/third_party/gaussian-splatting/SIBR_viewers/install"

if [[ -z "${DISPLAY:-}" ]]; then
  echo "Error: DISPLAY is not set. GUI forwarding needs X11/Wayland session."
  exit 1
fi

xhost +local:docker >/dev/null 2>&1 || true

DOCKER_RUN_FLAGS=(--rm)
if [[ -t 0 && -t 1 ]]; then
  DOCKER_RUN_FLAGS+=("-it")
fi

echo "[info] Launching SIBR viewer in container..."
${DOCKER_CMD} run "${DOCKER_RUN_FLAGS[@]}" \
  --gpus all \
  --network host \
  -e DISPLAY="${DISPLAY}" \
  -e QT_X11_NO_MITSHM=1 \
  -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace \
  "${IMAGE_NAME}" \
  bash -lc "${APP_PATH} -m '${CONTAINER_MODEL_PATH}' --appPath '${APP_ROOT}'"

echo "[done] SIBR viewer exited."
