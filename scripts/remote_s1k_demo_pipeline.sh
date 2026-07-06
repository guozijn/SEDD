#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-desktop-0f24dvl}"
REMOTE_DIR="${REMOTE_DIR:-~/Code/SEDD}"
PROXY="${PROXY:-http://172.27.0.1:7890}"
LIMIT="${LIMIT:-120}"
SEQ_LEN="${SEQ_LEN:-192}"
SFT_STEPS="${SFT_STEPS:-20}"
RL_UPDATES="${RL_UPDATES:-2}"
NUM_GENERATIONS="${NUM_GENERATIONS:-4}"
REPEAT_TIMES="${REPEAT_TIMES:-1}"
SAMPLE_STEPS="${SAMPLE_STEPS:-3}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-32}"
RL_MAX_NEW_TOKENS="${RL_MAX_NEW_TOKENS:-12}"

ssh "$HOST" "set -euo pipefail; cd $REMOTE_DIR; export PROXY='$PROXY'; source scripts/remote_env.sh; \
  bash scripts/setup_official_backend.sh; \
  mkdir -p runs/demo_models/base runs/demo_models/s1k_sft runs/demo_models/s1k_dcolt_rl runs/demo_models/evals data/processed; \
  uv run sedd-official-prepare --mode sft --hf-dataset simplescaling/s1K-1.1 --response-mode solution --seq-len $SEQ_LEN --limit $LIMIT --valid-ratio 0.15 --output-dir data/processed --name official_s1k; \
  uv run python - <<'PY'
import json
import re

from datasets import load_dataset


def compact(text, limit):
    text = re.sub(r'\s+', ' ', str(text or '')).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + '...'


rows = load_dataset('simplescaling/s1K-1.1', split='train')
items = []
for row in rows:
    question = str(row.get('question') or '').strip()
    solution = str(row.get('solution') or row.get('deepseek_attempt') or '').strip()
    if question and solution:
        items.append((str(row.get('id') or len(items)), question, solution))
    if len(items) >= 80:
        break
if len(items) < 2:
    raise RuntimeError('Need at least two S1K rows with question and solution fields.')

with open('data/processed/s1k_eval_prompts.txt', 'w', encoding='utf-8') as f:
    for _, question, _ in items[:40]:
        q = compact(question, 900)
        if q:
            f.write(q + '\n')

with open('data/processed/s1k_dcolt_records.jsonl', 'w', encoding='utf-8') as f:
    for idx, (source_id, question, solution) in enumerate(items[:40]):
        distractor = items[(idx + 1) % len(items)][2]
        correct_label = 'A' if idx % 2 == 0 else 'B'
        correct = compact(solution, 520)
        wrong = compact(distractor, 520)
        choices = {
            'A': correct if correct_label == 'A' else wrong,
            'B': wrong if correct_label == 'A' else correct,
        }
        prompt = (
            'Choose which candidate solution correctly solves the math problem. '
            'Return only the final choice as `Answer: <letter>`.\\n\\n'
            f'Problem: {compact(question, 700)}\\n'
            'Choices:\\n'
            f'A. {choices[\"A\"]}\\n'
            f'B. {choices[\"B\"]}'
        )
        f.write(json.dumps({
            'prompt': prompt,
            'answer': correct_label,
            'labels': ['A', 'B'],
            'source_id': source_id,
        }, ensure_ascii=False) + '\n')
PY
  uv run sedd-official-export-base --model-path louaaron/sedd-small --out runs/demo_models/base/checkpoint_base.pt --device cuda; \
  uv run sedd-official-eval --model-path runs/demo_models/base/checkpoint_base.pt --data data/processed/official_s1k_valid.pt --max-batches 10 --prompts-path data/processed/s1k_eval_prompts.txt --reward-samples 2 --seq-len $SEQ_LEN --sample-steps $SAMPLE_STEPS --max-new-tokens $MAX_NEW_TOKENS --device cuda | tee runs/demo_models/evals/base.jsonl; \
  uv run sedd-official-sft --model-path runs/demo_models/base/checkpoint_base.pt --train-path data/processed/official_s1k_train.pt --valid-path data/processed/official_s1k_valid.pt --out-dir runs/demo_models/s1k_sft --batch-size 1 --steps $SFT_STEPS --eval-every 10 --save-every 0 --log-every 5 --max-eval-batches 10 --device cuda; \
  uv run sedd-official-eval --model-path runs/demo_models/s1k_sft/checkpoint_last.pt --data data/processed/official_s1k_valid.pt --max-batches 10 --prompts-path data/processed/s1k_eval_prompts.txt --reward-samples 2 --seq-len $SEQ_LEN --sample-steps $SAMPLE_STEPS --max-new-tokens $MAX_NEW_TOKENS --device cuda | tee runs/demo_models/evals/sft.jsonl; \
  uv run sedd-official-rl --model-path runs/demo_models/s1k_sft/checkpoint_last.pt --reference-model-path runs/demo_models/base/checkpoint_base.pt --records-path data/processed/s1k_dcolt_records.jsonl --out-dir runs/demo_models/s1k_dcolt_rl --updates $RL_UPDATES --batch-size 1 --num-generations $NUM_GENERATIONS --repeat-times $REPEAT_TIMES --seq-len $SEQ_LEN --max-new-tokens $RL_MAX_NEW_TOKENS --sample-steps $SAMPLE_STEPS --save-every 0 --log-every 1 --device cuda; \
  uv run sedd-official-eval --model-path runs/demo_models/s1k_dcolt_rl/checkpoint_last.pt --data data/processed/official_s1k_valid.pt --max-batches 10 --prompts-path data/processed/s1k_eval_prompts.txt --reward-samples 2 --seq-len $SEQ_LEN --sample-steps $SAMPLE_STEPS --max-new-tokens $MAX_NEW_TOKENS --device cuda | tee runs/demo_models/evals/dcolt_rl.jsonl; \
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
      \"id\": \"dcolt_rl\",
      \"label\": \"S1K DCoLT RL\",
      \"backend\": \"official\",
      \"model_path\": \"runs/demo_models/s1k_dcolt_rl/checkpoint_last.pt\",
      \"description\": \"DCoLT-style exact-choice RL on MCQA verifier records derived from S1K\"
    }
  ]
}
JSON
  echo 'wrote runs/demo_models/registry.json'"
