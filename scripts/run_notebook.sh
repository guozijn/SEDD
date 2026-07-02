#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
JUPYTER_HOST="${JUPYTER_HOST:-127.0.0.1}"
JUPYTER_PORT="${JUPYTER_PORT:-8888}"

uv run jupyter lab \
  --no-browser \
  --ip "$JUPYTER_HOST" \
  --port "$JUPYTER_PORT" \
  notebooks/sedd_pipeline.ipynb
