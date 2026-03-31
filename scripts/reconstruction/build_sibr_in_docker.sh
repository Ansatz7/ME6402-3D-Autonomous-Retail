#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_NAME="${1:-sibr-builder:ubuntu22.04-cuda11.8}"
JOBS="${JOBS:-$(nproc)}"

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

if ! ${DOCKER_CMD} image inspect "${IMAGE_NAME}" >/dev/null 2>&1; then
  echo "Error: image not found: ${IMAGE_NAME}"
  echo "Run: bash scripts/reconstruction/build_sibr_docker_image.sh ${IMAGE_NAME}"
  exit 1
fi

echo "[info] Building SIBR in container..."
${DOCKER_CMD} run --rm \
  --gpus all \
  -v "${ROOT_DIR}:/workspace" \
  -w /workspace/third_party/gaussian-splatting/SIBR_viewers \
  "${IMAGE_NAME}" \
  bash -lc "git config --global --add safe.directory '*' || true && rm -rf build install extlibs/imgui/subbuild extlibs/imgui/src && find extlibs -type f -name CMakeCache.txt -delete && find extlibs -type d -name CMakeFiles -prune -exec rm -rf {} + && cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -G Ninja && cmake --build build -j ${JOBS} --target install"

echo "[done] SIBR build finished. Install dir: third_party/gaussian-splatting/SIBR_viewers/install"
