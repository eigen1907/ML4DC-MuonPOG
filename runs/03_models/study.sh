#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 caffeinate -i uv run --locked python -u scripts/study.py "$@"
