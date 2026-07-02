#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-desktop-0f24dvl}"
REMOTE_DIR="${REMOTE_DIR:-~/Code/SEDD}"
cd "$(dirname "$0")/.."
rsync -az \
  --exclude .git \
  --exclude .venv \
  --exclude data/processed \
  --exclude runs \
  --exclude checkpoints \
  --exclude external \
  --exclude __pycache__ \
  "$HOST:$REMOTE_DIR/" ./
echo "Synced $HOST:$REMOTE_DIR back to local repo"
