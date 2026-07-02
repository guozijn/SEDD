#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-desktop-0f24dvl}"
REMOTE_DIR="${REMOTE_DIR:-~/Code/SEDD}"
PROXY="${PROXY:-http://172.27.0.1:7890}"
LIMIT="${LIMIT:-120}"
SEQ_LEN="${SEQ_LEN:-192}"
SFT_STEPS="${SFT_STEPS:-20}"
RL_UPDATES="${RL_UPDATES:-2}"
SAMPLE_STEPS="${SAMPLE_STEPS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"

ssh "$HOST" "set -euo pipefail; cd $REMOTE_DIR; export PROXY='$PROXY'; source scripts/remote_env.sh; \
  bash scripts/setup_official_backend.sh; \
  mkdir -p runs/demo_models/base runs/demo_models/s1k_sft runs/demo_models/s1k_rl runs/demo_models/evals data/processed; \
  uv run sedd-official-prepare --mode sft --hf-dataset simplescaling/s1K-1.1 --response-mode solution --seq-len $SEQ_LEN --limit $LIMIT --valid-ratio 0.15 --output-dir data/processed --name official_s1k; \
  uv run python - <<'PY'
from datasets import load_dataset
rows = load_dataset('simplescaling/s1K-1.1', split='train')
with open('data/processed/s1k_rl_prompts.txt', 'w', encoding='utf-8') as f:
    for row in rows.select(range(min(40, len(rows)))):
        q = (row.get('question') or '').strip().replace('\n', ' ')
        if q:
            f.write(q + '\n')
PY
  uv run sedd-official-export-base --model-path louaaron/sedd-small --out runs/demo_models/base/checkpoint_base.pt --device cuda; \
  uv run sedd-official-eval --model-path runs/demo_models/base/checkpoint_base.pt --data data/processed/official_s1k_valid.pt --max-batches 10 --prompts-path data/processed/s1k_rl_prompts.txt --reward-samples 2 --seq-len $SEQ_LEN --sample-steps $SAMPLE_STEPS --max-new-tokens $MAX_NEW_TOKENS --device cuda | tee runs/demo_models/evals/base.jsonl; \
  uv run sedd-official-sft --model-path runs/demo_models/base/checkpoint_base.pt --train-path data/processed/official_s1k_train.pt --valid-path data/processed/official_s1k_valid.pt --out-dir runs/demo_models/s1k_sft --batch-size 1 --steps $SFT_STEPS --eval-every 10 --save-every $SFT_STEPS --log-every 5 --max-eval-batches 10 --device cuda; \
  uv run sedd-official-eval --model-path runs/demo_models/s1k_sft/checkpoint_last.pt --data data/processed/official_s1k_valid.pt --max-batches 10 --prompts-path data/processed/s1k_rl_prompts.txt --reward-samples 2 --seq-len $SEQ_LEN --sample-steps $SAMPLE_STEPS --max-new-tokens $MAX_NEW_TOKENS --device cuda | tee runs/demo_models/evals/sft.jsonl; \
  uv run sedd-official-rl --model-path runs/demo_models/s1k_sft/checkpoint_last.pt --reference-model-path runs/demo_models/base/checkpoint_base.pt --prompts-path data/processed/s1k_rl_prompts.txt --out-dir runs/demo_models/s1k_rl --updates $RL_UPDATES --batch-size 1 --seq-len $SEQ_LEN --max-new-tokens $MAX_NEW_TOKENS --sample-steps $SAMPLE_STEPS --save-every $RL_UPDATES --device cuda; \
  uv run sedd-official-eval --model-path runs/demo_models/s1k_rl/checkpoint_last.pt --data data/processed/official_s1k_valid.pt --max-batches 10 --prompts-path data/processed/s1k_rl_prompts.txt --reward-samples 2 --seq-len $SEQ_LEN --sample-steps $SAMPLE_STEPS --max-new-tokens $MAX_NEW_TOKENS --device cuda | tee runs/demo_models/evals/rl.jsonl; \
  cat > runs/demo_models/registry.json <<'JSON'
{
  \"default_model_id\": \"sft\",
  \"models\": [
    {
      \"id\": \"base\",
      \"label\": \"SEDD small base\",
      \"backend\": \"official\",
      \"model_path\": \"runs/demo_models/base/checkpoint_base.pt\",
      \"description\": \"Original louaaron/sedd-small exported locally\"
    },
    {
      \"id\": \"sft\",
      \"label\": \"S1K SFT\",
      \"backend\": \"official\",
      \"model_path\": \"runs/demo_models/s1k_sft/checkpoint_last.pt\",
      \"description\": \"Official small SFT on simplescaling/s1K-1.1 solution targets\"
    },
    {
      \"id\": \"rl\",
      \"label\": \"S1K SFT + RL\",
      \"backend\": \"official\",
      \"model_path\": \"runs/demo_models/s1k_rl/checkpoint_last.pt\",
      \"description\": \"Heuristic reward post-training from the S1K SFT checkpoint\"
    }
  ]
}
JSON
  echo 'wrote runs/demo_models/registry.json'"
