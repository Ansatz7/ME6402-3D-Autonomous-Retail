#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE_NAME="${1:-sibr-builder:ubuntu22.04-cuda11.8}"

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

echo "[info] Building image: ${IMAGE_NAME}"
${DOCKER_CMD} build \
  -t "${IMAGE_NAME}" \
  -f "${ROOT_DIR}/docker/sibr/Dockerfile" \
  "${ROOT_DIR}"

echo "[done] Image built: ${IMAGE_NAME}"
