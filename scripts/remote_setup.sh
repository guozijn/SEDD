#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-desktop-0f24dvl}"
REMOTE_DIR="${REMOTE_DIR:-~/Code/SEDD}"
PROXY="${PROXY:-http://172.27.0.1:7890}"
ssh "$HOST" "set -euo pipefail; cd $REMOTE_DIR; export PROXY='$PROXY'; source scripts/remote_env.sh; command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh; if [ ! -d .venv ]; then uv venv --python 3.12 || uv venv; fi; uv sync --extra dev"
