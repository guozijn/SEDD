#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv venv --python 3.12
uv sync --extra dev
echo "Environment ready. Use: source .venv/bin/activate"
