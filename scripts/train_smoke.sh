#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run sedd-prepare --mode pretrain --seq-len 128 --output-dir data/processed --name pretrain
uv run sedd-prepare --mode sft --seq-len 128 --output-dir data/processed --name sft
cat > data/processed/rl_prompts.txt <<'EOF'
Explain why score entropy is useful for discrete diffusion.
Describe one challenge when adapting RLHF to SEDD.
EOF
uv run sedd-train --config configs/tiny_pretrain.yaml train.steps=3 train.eval_every=3 train.save_every=3 train.out_dir=runs/smoke_pretrain train.batch_size=2
uv run sedd-train --config configs/tiny_sft.yaml train.resume=runs/smoke_pretrain/checkpoint_last.pt train.steps=3 train.eval_every=3 train.save_every=3 train.out_dir=runs/smoke_sft train.batch_size=2
uv run sedd-eval --checkpoint runs/smoke_sft/checkpoint_last.pt --data data/processed/sft_valid.pt --batch-size 2 --max-batches 2
uv run sedd-sample --checkpoint runs/smoke_sft/checkpoint_last.pt --prompt "Explain SEDD briefly." --max-new-tokens 32 --steps 4
