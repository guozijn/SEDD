#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p external
if [ ! -d external/Score-Entropy-Discrete-Diffusion/.git ]; then
  git clone https://github.com/louaaron/Score-Entropy-Discrete-Diffusion.git external/Score-Entropy-Discrete-Diffusion
else
  git -C external/Score-Entropy-Discrete-Diffusion pull --ff-only
fi

uv sync --extra dev --extra official --extra datasets

cat <<'MSG'
Official backend source/deps are prepared.

The upstream model is fastest with flash-attn on CUDA. This project includes a
PyTorch SDPA fallback for validation if flash-attn is not available. For faster
remote runs, install a flash-attn build compatible with that machine's
PyTorch/CUDA, for example:

  uv pip install flash-attn --no-build-isolation

Then check loading with:

  uv run sedd-official-check --model-path louaaron/sedd-small --device cuda

MSG
