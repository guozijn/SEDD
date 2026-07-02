#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MODEL_PATH="${MODEL_PATH:-louaaron/sedd-small}"
DEVICE="${DEVICE:-auto}"
uv run sedd-official-prepare --mode sft --seq-len "${SEQ_LEN:-128}" --output-dir data/processed --name official_sft
uv run sedd-official-sft \
  --model-path "$MODEL_PATH" \
  --train-path data/processed/official_sft_train.pt \
  --valid-path data/processed/official_sft_valid.pt \
  --out-dir runs/official_sft_smoke \
  --batch-size 1 \
  --steps "${STEPS:-1}" \
  --eval-every "${EVAL_EVERY:-1}" \
  --save-every "${SAVE_EVERY:-1}" \
  --device "$DEVICE"
