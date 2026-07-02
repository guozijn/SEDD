# Runbook

## Local Setup

```bash
scripts/setup_uv.sh
source .venv/bin/activate
```

## Data Preparation

Toy data:

```bash
scripts/prepare_toy.sh
```

Custom pretraining text:

```bash
uv run sedd-prepare \
  --mode pretrain \
  --input data/raw/my_texts \
  --seq-len 128 \
  --output-dir data/processed \
  --name pretrain
```

Custom SFT JSONL:

```bash
uv run sedd-prepare \
  --mode sft \
  --input data/raw/sft.jsonl \
  --seq-len 128 \
  --output-dir data/processed \
  --name sft
```

JSONL rows can use `prompt`/`response`, `instruction`/`output`, or
`question`/`answer`.

## Train

```bash
uv run sedd-train --config configs/tiny_pretrain.yaml
uv run sedd-train --config configs/tiny_sft.yaml
```

Override any config value with dotted syntax:

```bash
uv run sedd-train --config configs/tiny_pretrain.yaml train.steps=100 train.batch_size=4
```

## RL Post-Training

```bash
uv run sedd-rl --config configs/tiny_rl.yaml
```

Edit `configs/tiny_rl.yaml` to point at a different checkpoint, prompts file, or
reward function.

## Evaluate and Sample

```bash
uv run sedd-eval --checkpoint runs/sft_tiny/checkpoint_last.pt --data data/processed/sft_valid.pt

uv run sedd-sample \
  --checkpoint runs/sft_tiny/checkpoint_last.pt \
  --prompt "Explain score entropy in SEDD."

uv run sedd-sample \
  --checkpoint runs/sft_tiny/checkpoint_last.pt \
  --infill "SEDD can infill text because [MASK]."
```

## Demo App

```bash
BACKEND=official MODEL_PATH=louaaron/sedd-small DEVICE=cuda scripts/run_app.sh
```

Open `http://127.0.0.1:8000`.

Mini backend app:

```bash
BACKEND=mini scripts/run_app.sh runs/sft_tiny/checkpoint_last.pt
```

## Remote Workflow

Sync code to the remote:

```bash
scripts/remote_sync_to.sh
```

Install the remote environment:

```bash
scripts/remote_setup.sh
```

Run smoke test on remote:

```bash
scripts/remote_smoke.sh
```

Run a longer GPU pretraining job:

```bash
scripts/remote_train_16gb.sh
```

Official checkpoint sampling:

```bash
scripts/remote_setup_official.sh
scripts/remote_official_sample.sh
```

Official SFT and RL post-training:

```bash
scripts/official_prepare_toy.sh
DEVICE=cuda MODEL_PATH=louaaron/sedd-small scripts/official_sft_smoke.sh
DEVICE=cuda \
  MODEL_PATH=runs/official_sft_smoke/checkpoint_last.pt \
  REFERENCE_MODEL_PATH=louaaron/sedd-small \
  scripts/official_rl_smoke.sh
```

One remote command for the whole official path:

```bash
scripts/remote_official_pipeline_smoke.sh
```

ARC base/SFT/RL frontend comparison:

```text
runs/arc_models/base/checkpoint_base.pt
runs/arc_models/arc_sft/checkpoint_last.pt
runs/arc_models/arc_rl/checkpoint_last.pt
```

Copy the registry template after syncing or training:

```bash
cp configs/arc_model_registry.json runs/arc_models/registry.json
```

`scripts/run_app.sh` auto-detects `runs/arc_models/registry.json` when it exists:

```bash
scripts/run_app.sh
```

If the model dropdown only shows one option, restart the app from the repo root
after confirming `runs/arc_models/registry.json` exists, or pass
`MODEL_REGISTRY=runs/arc_models/registry.json` explicitly.

Evaluation interpretation:

- Base vs ARC SFT: compare ARC validation `score_entropy`; lower is better.
- ARC SFT vs ARC RL: compare both `score_entropy` and exact-answer reward; RL may
  improve the chosen reward while slightly hurting validation score entropy.

Official fine-tuned checkpoints are standard `.pt` files with the base HF model
path embedded. They can be served with:

```bash
BACKEND=official MODEL_PATH=runs/official_rl_smoke/checkpoint_last.pt DEVICE=cuda scripts/run_app.sh
```

Sync code back after remote edits or generated docs:

```bash
scripts/remote_sync_back.sh
```

The default sync-back excludes `runs/`, `checkpoints/`, `.venv/`, and
`data/processed/` to avoid copying large artifacts.

## Demo-Day Checklist

1. Show `docs/sedd_understanding.md` and explain the ratio objective.
2. Show `src/sedd_mini/diffusion.py` for the score entropy implementation.
3. Show `src/sedd_mini/train.py` for pretraining/SFT reuse.
4. Show `src/sedd_mini/posttrain_rl.py` for the RL adaptation.
5. Show `src/sedd_mini/official_finetune.py` and `src/sedd_mini/official_posttrain_rl.py`.
6. Run `sedd-sample` with the default official backend, or `--backend mini` for the compact model.
7. Be explicit about limitations and the upgrade path to LoRA/SEPO.
