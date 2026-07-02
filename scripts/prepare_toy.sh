#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run sedd-prepare --mode pretrain --seq-len 128 --output-dir data/processed --name pretrain
uv run sedd-prepare --mode sft --seq-len 128 --output-dir data/processed --name sft
cat > data/processed/rl_prompts.txt <<'EOF'
Explain why score entropy is useful for discrete diffusion.
Describe one challenge when adapting RLHF to SEDD.
Give a concise answer about bidirectional attention in diffusion language models.
EOF
