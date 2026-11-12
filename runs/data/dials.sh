#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

UV_PROJECT_ENVIRONMENT=.venv-dials uv run --managed-python --locked --only-group dials python -u -m scripts.data.fetch_dials "$@"
