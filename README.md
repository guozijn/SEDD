# SEDD Pipeline Project

This repo is a compact, runnable implementation of a Score Entropy Discrete
Diffusion language-modeling pipeline. It contains both a small byte-level
implementation for end-to-end experimentation and an official SEDD backend for
`louaaron/sedd-small` / `louaaron/sedd-medium`.

It includes:

- TinyStories mini pretraining and a compact mini SFT continuation.
- A bidirectional byte-level Transformer score network.
- Absorbing-state score entropy training loss with log-linear noise.
- Prompt-conditioned SFT by keeping prompt tokens visible and applying loss only
  to answer tokens.
- Official SEDD LoRA SFT on ARC-Easy, saved as a merged serving checkpoint.
- Official SEDD DCoLT-style grouped RL on ARC-Challenge with group-normalized
  advantages, PPO-style clipping, and a frozen reference KL term.
- Evaluation, sampling, infilling, FastAPI backend, and browser frontend.
- Frontend model switching for mini, official base, official medium, LoRA SFT,
  and DCoLT RL checkpoints.
- Remote `uv` and `rsync` scripts for `desktop-0f24dvl`.

The implementation is intentionally small enough to smoke-test quickly, while the
interfaces are the same ones needed for longer training on a 16 GB GPU.

## Quick Start

```bash
scripts/setup_uv.sh
scripts/prepare_toy.sh
uv run sedd-train --config configs/tiny_pretrain.yaml
uv run sedd-train --config configs/tiny_sft.yaml
uv run sedd-eval --checkpoint runs/sft_tiny/checkpoint_last.pt --data data/processed/sft_valid.pt
uv run sedd-sample --backend mini --checkpoint runs/sft_tiny/checkpoint_last.pt --prompt "Explain SEDD briefly."
```

For the notebook walkthrough, including TinyStories compact pretraining, mini
SFT, official ARC-Easy LoRA SFT, official ARC-Challenge DCoLT RL, evaluation,
and inference:

```bash
uv sync --extra notebook --extra official --extra datasets
scripts/run_notebook.sh
```

The committed notebook [notebooks/sedd_pipeline.ipynb](notebooks/sedd_pipeline.ipynb)
is an executed version. The last remote run used:

```bash
FRESH_RUN=1 \
DEVICE=cuda \
MINI_PRETRAIN_STEPS=2000 \
MINI_SFT_STEPS=50 \
OFFICIAL_SFT_STEPS=1000 \
RL_UPDATES=100 \
SAMPLE_STEPS=4 \
MAX_NEW_TOKENS=12 \
uv run --extra notebook jupyter nbconvert \
  --to notebook \
  --execute notebooks/sedd_pipeline.ipynb \
  --output sedd_pipeline.ipynb \
  --output-dir notebooks \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3
```

If launching Jupyter on `desktop-0f24dvl` from your Mac, tunnel the notebook port:

```bash
ssh -L 8888:127.0.0.1:8888 desktop-0f24dvl
```

Run the default registry-backed demo on the remote GPU:

```bash
cp configs/arc_model_registry.json runs/arc_models/registry.json
BACKEND=official DEVICE=auto scripts/run_app.sh
```

Open `http://127.0.0.1:8000`.

The current frontend registry exposes:

- `base`: exported official `louaaron/sedd-small`.
- `medium`: official `louaaron/sedd-medium` from the local HF cache.
- `arc_lora_sft`: official SEDD-small LoRA SFT on ARC-Easy.
- `arc_dcolt_rl`: official SEDD-small DCoLT-style RL on ARC-Challenge.

The default frontend model is `arc_lora_sft`; the dropdown intentionally lists
only official SEDD checkpoints.

## Remote Workflow

```bash
scripts/remote_sync_to.sh
scripts/remote_setup.sh
scripts/remote_smoke.sh
scripts/remote_sync_back.sh
```

For a longer 16 GB GPU run:

```bash
scripts/remote_train_16gb.sh
```

For the S1K remote demo pipeline:

```bash
scripts/remote_s1k_demo_pipeline.sh
```

This exports `louaaron/sedd-small`, runs SFT on S1K solution targets, builds
`data/processed/s1k_dcolt_records.jsonl` as exact-choice DCoLT verifier records
derived from S1K, runs `sedd-official-rl`, and writes
`runs/demo_models/registry.json` with `base`, `sft`, and `dcolt_rl` entries.

The remote is WSL2. If `nvidia-smi` is not on `PATH`, the script checks
`/usr/lib/wsl/lib/nvidia-smi`.

## Official SEDD Backend

The project has two model backends. The user-facing default is `official`:

- `mini`: this repo's compact byte-level model for pretrain/SFT demos.
- `official`: the upstream SEDD architecture and HF checkpoints such as
  `louaaron/sedd-small` and `louaaron/sedd-medium`.

Prepare the official backend:

```bash
scripts/setup_official_backend.sh
```

The upstream model uses `flash-attn`, so official sampling is expected to run on
CUDA, not local Apple MPS/CPU. On the remote GPU:

```bash
scripts/remote_setup_official.sh
scripts/remote_official_sample.sh
```

Direct CLI usage:

```bash
uv run sedd-sample \
  --backend official \
  --model-path louaaron/sedd-small \
  --device cuda \
  --prompt "Explain score entropy in one paragraph." \
  --steps 32
```

Serve the frontend against the official checkpoint:

```bash
BACKEND=official MODEL_PATH=louaaron/sedd-small DEVICE=cuda scripts/run_app.sh
```

The canonical LoRA SFT and DCoLT RL path is the executed notebook. After the
notebook data-preparation cells have produced the ARC tensors/records, the same
steps can be rerun through the aligned CLIs:

```bash
uv run sedd-official-export-base \
  --model-path louaaron/sedd-small \
  --out runs/arc_models/base/checkpoint_base.pt \
  --device cuda

uv run sedd-official-sft --device cuda
uv run sedd-official-rl --device cuda
```

These defaults match the notebook artifact layout:

- `sedd-official-sft` reads `data/processed/official_arc_easy_train.pt` and
  writes `runs/arc_models/arc_lora_sft/checkpoint_last.pt`.
- `sedd-official-rl` reads `data/processed/arc_challenge_rl_train.jsonl`, starts
  from `runs/arc_models/arc_lora_sft/checkpoint_last.pt`, and writes
  `runs/arc_models/arc_dcolt_rl/checkpoint_last.pt`.

The notebook and remote run produce one current checkpoint per model version:

- `runs/notebook_tinystories_pretrain/checkpoint_last.pt`
- `runs/notebook_sft/checkpoint_last.pt`
- `runs/arc_models/base/checkpoint_base.pt`
- `runs/arc_models/arc_lora_sft/checkpoint_last.pt`
- `runs/arc_models/arc_dcolt_rl/checkpoint_last.pt`

Copy `configs/arc_model_registry.json` to `runs/arc_models/registry.json` after
training or syncing. `scripts/run_app.sh` auto-detects this ARC registry when it
exists, so this is enough on the remote demo machine:

```bash
scripts/run_app.sh
```

You can still pass `MODEL_REGISTRY=runs/arc_models/registry.json` explicitly if
you want to point the app at the ARC registry path.

Use validation score entropy for ARC SFT quality and exact-answer reward logs
for ARC RL behavior; the short default run is demonstrative, not converged.

## Project Map

- `src/sedd_mini/diffusion.py`: absorbing forward process and score entropy loss.
- `src/sedd_mini/model.py`: bidirectional Transformer score model.
- `src/sedd_mini/train.py`: pretraining and SFT loop.
- `src/sedd_mini/evaluate.py`: score entropy, denoising CE, pseudo-perplexity.
- `src/sedd_mini/sampling.py`: prompt generation and infilling sampler.
- `src/sedd_mini/server.py`: backend for the frontend demo.
- `src/sedd_mini/official_backend.py`: optional adapter for official SEDD HF checkpoints.
- `src/sedd_mini/official_prepare_data.py`: GPT-2-token data preparation for official SEDD.
- `src/sedd_mini/official_finetune.py`: LoRA response-only score-entropy SFT for official SEDD.
- `src/sedd_mini/official_posttrain_rl.py`: DCoLT-style grouped RL for official SEDD.
- `src/sedd_mini/mcqa_data.py`: ARC multiple-choice formatting and exact-answer reward.
- `configs/`: tiny and remote training configs.
- `notebooks/sedd_pipeline.ipynb`: executed end-to-end pipeline notebook.

## References

- Paper: https://arxiv.org/abs/2310.16834
- Official SEDD repo: https://github.com/louaaron/Score-Entropy-Discrete-Diffusion
- `louaaron/sedd-small`: https://huggingface.co/louaaron/sedd-small
- `louaaron/sedd-medium`: https://huggingface.co/louaaron/sedd-medium
- LLaDOU / DCoLT reference: https://github.com/maple-research-lab/LLaDOU
