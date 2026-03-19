#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
THIRD_PARTY_DIR="$ROOT_DIR/third_party"
GS_DIR="$THIRD_PARTY_DIR/gaussian-splatting"

mkdir -p "$THIRD_PARTY_DIR"

if [[ ! -d "$GS_DIR/.git" ]]; then
  echo "[setup] Cloning gaussian-splatting into $GS_DIR"
  git clone https://github.com/graphdeco-inria/gaussian-splatting.git "$GS_DIR"
else
  echo "[setup] gaussian-splatting already exists: $GS_DIR"
fi

echo "[setup] Done. Next steps:"
echo "  1) Create and activate a Python env"
echo "  2) pip install -r $GS_DIR/requirements.txt"
echo "  3) Build required extensions as documented by upstream repo"
