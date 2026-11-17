#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Optional model comparison; the default dataset is the main 1 pb^-1 grouping.
if [[ $# -eq 0 ]]; then set -- ls_group_1pb; fi
for scope in "$@"; do
    case "$scope" in
        ls_group_1000glb|ls_group_7pb|ls_group_1pb|ls_no_group) ;;
        *) printf 'Unknown LS version: %s\n' "$scope" >&2; exit 2 ;;
    esac
    options=(--years 2024 2025 2026 --ls-mode "$scope")
    for model in model_01_ae_2feat model_02_ae_gmm_mse model_03_ae_gmm_deviance model_04_ae_gmm_deviance_norm_me_bin; do
        name=${model#model_}
        bash runs/train.sh "$model" "${options[@]}"
        bash runs/predict.sh "output/model/$scope/$name" data/dials "output/predict/$scope/$name" \
            --years 2024 2025 2026 --prompt-reco --reuse
        if [[ "$name" == *_ae_gmm* ]]; then
            uv run --locked python -m scripts.evaluate_gmm --input "output/predict/$scope/$name"
            bash runs/plotting/components.sh --input "output/predict/$scope/$name"
        else
            bash runs/plotting/scores.sh --input "output/predict/$scope/$name"
        fi
    done
done
