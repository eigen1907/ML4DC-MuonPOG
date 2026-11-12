#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -lt 2 ]]; then
    printf 'Usage: bash runs/predict.sh CHECKPOINT DATA [OUTPUT] [OPTIONS]\n' >&2
    exit 2
fi
args=(--checkpoint "$1" --data "$2")
shift 2
if [[ $# -gt 0 && "$1" != --* ]]; then
    args+=(--output "$1")
    shift
fi
exec uv run --locked python -m scripts.predict "${args[@]}" "$@"
