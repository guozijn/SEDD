#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
MODEL_PATH="${MODEL_PATH:-runs/official_sft_smoke/checkpoint_last.pt}"
REFERENCE_MODEL_PATH="${REFERENCE_MODEL_PATH:-louaaron/sedd-small}"
DEVICE="${DEVICE:-auto}"
cat > data/processed/rl_prompts.txt <<'EOF'
Explain why score entropy is useful for discrete diffusion.
Describe one challenge when adapting RLHF to SEDD.
EOF
uv run sedd-official-rl \
  --model-path "$MODEL_PATH" \
  --reference-model-path "$REFERENCE_MODEL_PATH" \
  --prompts-path data/processed/rl_prompts.txt \
  --out-dir runs/official_rl_smoke \
  --updates "${UPDATES:-1}" \
  --batch-size 1 \
  --seq-len "${SEQ_LEN:-128}" \
  --max-new-tokens "${MAX_NEW_TOKENS:-16}" \
  --sample-steps "${SAMPLE_STEPS:-2}" \
  --device "$DEVICE"
