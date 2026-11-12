#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

metadata_groups=(--only-group metadata)
if [[ " $* " != *" --offline "* ]]; then
  metadata_groups+=(--only-group kerberos)
fi
UV_PROJECT_ENVIRONMENT=.venv-metadata uv run --managed-python --locked "${metadata_groups[@]}" python scripts/fetch_rr.py "$@"
