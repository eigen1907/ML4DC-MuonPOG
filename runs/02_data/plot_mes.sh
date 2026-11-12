#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

uv run --locked python -m scripts.plotting.mes "$@"
