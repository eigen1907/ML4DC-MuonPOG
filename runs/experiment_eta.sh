#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

exec uv run --locked python -u -m scripts.experiment_eta
