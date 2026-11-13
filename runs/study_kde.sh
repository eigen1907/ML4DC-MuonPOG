#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Cached data only; existing compatible training checkpoints are preserved.
bash runs/train.sh model_07_ae_kde_adaptive --years 2024 2025
bash runs/train.sh model_08_ae_kde_masked --years 2024 2025
exec uv run --locked python -u -m scripts.study_kde
