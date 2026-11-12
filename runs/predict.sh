#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -lt 2 || $# -gt 3 ]]; then
    printf 'Usage: bash runs/predict.sh CHECKPOINT DATA [OUTPUT]\n' >&2
    exit 2
fi
args=(--checkpoint "$1" --data "$2")
if [[ $# -eq 3 ]]; then
    args+=(--output "$3")
fi
uv run --locked python -m scripts.predict "${args[@]}"
