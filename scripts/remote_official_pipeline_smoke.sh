#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-desktop-0f24dvl}"
REMOTE_DIR="${REMOTE_DIR:-~/Code/SEDD}"
MODEL_PATH="${MODEL_PATH:-louaaron/sedd-small}"
PROXY="${PROXY:-http://172.27.0.1:7890}"
ssh "$HOST" "set -euo pipefail; cd $REMOTE_DIR; export PROXY='$PROXY'; source scripts/remote_env.sh; bash scripts/setup_official_backend.sh; DEVICE=cuda MODEL_PATH=$MODEL_PATH STEPS=1 SEQ_LEN=128 bash scripts/official_sft_smoke.sh; DEVICE=cuda MODEL_PATH=runs/official_sft_smoke/checkpoint_last.pt REFERENCE_MODEL_PATH=$MODEL_PATH UPDATES=1 SAMPLE_STEPS=2 MAX_NEW_TOKENS=16 SEQ_LEN=128 bash scripts/official_rl_smoke.sh; uv run sedd-sample --backend official --model-path runs/official_rl_smoke/checkpoint_last.pt --device cuda --prompt \"Explain SEDD briefly.\" --steps 2"
