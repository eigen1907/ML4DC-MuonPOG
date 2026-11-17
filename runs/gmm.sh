#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Main experiment. Pass another LS mode only for an explicit comparison.
if [[ $# -eq 0 ]]; then set -- ls_group_1pb; fi
name=03_ae_gmm_deviance
for scope in "$@"; do
    case "$scope" in
        ls_group_1000glb|ls_group_7pb|ls_group_1pb|ls_no_group) ;;
        *) printf 'Unknown LS version: %s\n' "$scope" >&2; exit 2 ;;
    esac
    options=(--years 2024 2025 2026 --ls-mode "$scope")
    bash runs/train.sh "model_$name" "${options[@]}"
    bash runs/predict.sh "output/model/$scope/$name" data/dials "output/predict/$scope/$name" \
        --years 2024 2025 2026 --prompt-reco --reuse
    uv run --locked python -m scripts.evaluate_gmm --input "output/predict/$scope/$name"
    bash runs/plotting/components.sh --input "output/predict/$scope/$name"
done
