#!/usr/bin/env bash
set -euo pipefail

# Ubuntu 20.04/22.04: install Docker Engine + NVIDIA Container Toolkit.
# Must run with sudo privileges.

if [[ "${EUID}" -ne 0 ]]; then
  echo "Error: please run as root (sudo bash $0)"
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "Error: apt-get not found. This installer targets Ubuntu/Debian."
  exit 1
fi

echo "[1/7] Installing base packages..."
apt-get update
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  gnupg \
  lsb-release \
  software-properties-common

echo "[2/7] Installing Docker Engine..."
apt-get install -y --no-install-recommends docker.io
systemctl enable docker
systemctl restart docker

echo "[3/7] Configuring NVIDIA Container Toolkit apt source..."
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  gpg --dearmor -o /etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg

curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/etc/apt/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list

echo "[4/7] Installing NVIDIA Container Toolkit..."
apt-get update
apt-get install -y --no-install-recommends nvidia-container-toolkit

echo "[5/7] Configuring Docker runtime for NVIDIA..."
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "[6/7] Post-install note for current user..."
if [[ -n "${SUDO_USER:-}" ]]; then
  usermod -aG docker "${SUDO_USER}" || true
  echo "Added ${SUDO_USER} to docker group."
  echo "Please re-login (or run: newgrp docker) before using docker without sudo."
else
  echo "SUDO_USER not detected. Add your user to docker group manually if needed:"
  echo "  sudo usermod -aG docker <your_user>"
fi

echo "[7/7] Verifying installation..."
docker --version
nvidia-smi || true

echo "Testing GPU access from container..."
docker run --rm --gpus all nvidia/cuda:11.8.0-runtime-ubuntu22.04 nvidia-smi

echo "[done] Docker + NVIDIA container runtime installed and verified."
