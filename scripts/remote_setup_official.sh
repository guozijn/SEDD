#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-desktop-0f24dvl}"
REMOTE_DIR="${REMOTE_DIR:-~/Code/SEDD}"
PROXY="${PROXY:-http://172.27.0.1:7890}"
ssh "$HOST" "set -euo pipefail; cd $REMOTE_DIR; export PROXY='$PROXY'; source scripts/remote_env.sh; bash scripts/setup_official_backend.sh"
