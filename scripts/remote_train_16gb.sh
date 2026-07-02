#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-desktop-0f24dvl}"
REMOTE_DIR="${REMOTE_DIR:-~/Code/SEDD}"
CONFIG="${CONFIG:-configs/remote_16gb_pretrain.yaml}"
PROXY="${PROXY:-http://172.27.0.1:7890}"
ssh "$HOST" "set -euo pipefail; cd $REMOTE_DIR; export PROXY='$PROXY'; source scripts/remote_env.sh; if [ -x /usr/lib/wsl/lib/nvidia-smi ]; then /usr/lib/wsl/lib/nvidia-smi; fi; uv run sedd-train --config $CONFIG"
