#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run sedd-official-prepare --mode sft --seq-len "${SEQ_LEN:-128}" --output-dir data/processed --name official_sft
cat > data/processed/rl_prompts.txt <<'EOF'
Explain why score entropy is useful for discrete diffusion.
Describe one challenge when adapting RLHF to SEDD.
Give a concise answer about bidirectional attention in diffusion language models.
EOF
