#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
CHECKPOINT="${1:-runs/sft_tiny/checkpoint_last.pt}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
BACKEND="${BACKEND:-official}"
MODEL_PATH="${MODEL_PATH:-louaaron/sedd-small}"
OFFICIAL_REPO="${OFFICIAL_REPO:-external/Score-Entropy-Discrete-Diffusion}"
MODEL_REGISTRY="${MODEL_REGISTRY:-}"
DEFAULT_MODEL_ID="${DEFAULT_MODEL_ID:-}"
ARC_REGISTRY="runs/arc_models/registry.json"
UV_BIN="${UV_BIN:-$(command -v uv || true)}"

if [ -z "$UV_BIN" ] && [ -x "$HOME/.local/bin/uv" ]; then
  UV_BIN="$HOME/.local/bin/uv"
fi

if [ -z "$UV_BIN" ]; then
  echo "uv not found. Set UV_BIN=/path/to/uv or install uv." >&2
  exit 127
fi

if [ "$BACKEND" = "official" ] && [ -z "$MODEL_REGISTRY" ] && [ -f "$ARC_REGISTRY" ]; then
  MODEL_REGISTRY="$ARC_REGISTRY"
fi

if [ "$BACKEND" = "official" ]; then
  if [ -n "$MODEL_REGISTRY" ]; then
    echo "Serving official models from registry: $MODEL_REGISTRY"
    "$UV_BIN" run sedd-api --backend official --model-registry "$MODEL_REGISTRY" --default-model-id "$DEFAULT_MODEL_ID" --official-repo "$OFFICIAL_REPO" --host "$HOST" --port "$PORT" --device "${DEVICE:-auto}"
  else
    echo "Serving single official model: $MODEL_PATH"
    "$UV_BIN" run sedd-api --backend official --model-path "$MODEL_PATH" --official-repo "$OFFICIAL_REPO" --host "$HOST" --port "$PORT" --device "${DEVICE:-auto}"
  fi
else
  echo "Serving mini checkpoint: $CHECKPOINT"
  "$UV_BIN" run sedd-api --backend mini --checkpoint "$CHECKPOINT" --host "$HOST" --port "$PORT" --device "${DEVICE:-auto}"
fi
