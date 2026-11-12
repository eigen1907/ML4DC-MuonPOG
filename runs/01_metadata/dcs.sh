#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

UV_PROJECT_ENVIRONMENT=.venv-metadata uv run --managed-python --locked --only-group metadata python scripts/fetch_metadata.py dcs "$@"
