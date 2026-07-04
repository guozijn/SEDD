#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MODEL_PATH="${MODEL_PATH:-runs/official_sft_smoke/checkpoint_last.pt}"
REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH:-louaaron/sedd-small}"
DEVICE="${DEVICE:-auto}"
cat > data/processed/arc_rl_smoke.jsonl <<'EOF'
{"prompt":"Answer the science multiple-choice question. Return only the final choice as `Answer: <letter>`.\n\nQuestion: Which object is most likely to conduct electricity?\nChoices:\nA. copper wire\nB. rubber band\nC. wooden spoon\nD. plastic cup","answer":"A","labels":["A","B","C","D"],"source_id":"smoke-1"}
EOF
uv run sedd-official-rl \
  --model-path "$MODEL_PATH" \
  --reference-model-path "$REFERENCE_MODEL_PATH" \
  --records-path data/processed/arc_rl_smoke.jsonl \
  --out-dir runs/official_rl_smoke \
  --updates "${UPDATES:-1}" \
  --batch-size 1 \
  --num-generations "${NUM_GENERATIONS:-2}" \
  --seq-len "${SEQ_LEN:-128}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
  --sample-steps "${SAMPLE_STEPS:-2}" \
  --device "$DEVICE"
