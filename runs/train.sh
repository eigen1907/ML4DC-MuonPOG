#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -lt 1 ]]; then
    printf 'Usage: bash runs/train.sh MODEL [OPTIONS]\n' >&2
    exit 2
fi
model=$1
shift
exec uv run --locked python -u -m scripts.train --model "$model" "$@"
