#!/usr/bin/env bash

export PATH="$HOME/.local/bin:$PATH"
if [ -n "${PROXY:-}" ]; then
  export HTTP_PROXY="$PROXY"
  export HTTPS_PROXY="$PROXY"
  export ALL_PROXY="$PROXY"
  export http_proxy="$PROXY"
  export https_proxy="$PROXY"
  export all_proxy="$PROXY"
fi
