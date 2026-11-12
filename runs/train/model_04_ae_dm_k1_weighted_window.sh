#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

uv run --locked python -u -m scripts.train --model model_04_ae_dm_k1_weighted_window "$@"
