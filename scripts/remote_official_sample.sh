#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-desktop-0f24dvl}"
REMOTE_DIR="${REMOTE_DIR:-~/Code/SEDD}"
MODEL_PATH="${MODEL_PATH:-louaaron/sedd-small}"
PROMPT="${PROMPT:-Explain score entropy in one paragraph.}"
STEPS="${STEPS:-32}"
PROXY="${PROXY:-http://172.27.0.1:7890}"
ssh "$HOST" "set -euo pipefail; cd $REMOTE_DIR; export PROXY='$PROXY'; source scripts/remote_env.sh; uv run sedd-sample --backend official --model-path $MODEL_PATH --device cuda --prompt \"$PROMPT\" --steps $STEPS"
