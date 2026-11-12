#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
mkdir -p output/logs
exec > >(tee -a output/logs/study.log) 2>&1
printf 'running\n' > output/logs/study.status
trap 'study_status=$?; printf "exit_code=%s\n" "$study_status" > output/logs/study.status' EXIT
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 caffeinate -i uv run --locked python -u scripts/study.py "$@"
