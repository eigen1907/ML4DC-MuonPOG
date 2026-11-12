#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

uv run --locked python -u -m scripts.train --model model_03_ae_dm_k1 "$@"
