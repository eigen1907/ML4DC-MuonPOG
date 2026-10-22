#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

index=output/plots/mes/plots.md
rm -rf output/plots/mes
mkdir -p "$(dirname "$index")"
printf '# Raw Muons ME plots\n\nThe additional Good test runs have the three highest ae_bin flagged-LS fractions among Good test runs with at least 100 LS (score > 1).\n\n' > "$index"

for folder in \
  output/dataset/raw/Muon0__Run2025F-PromptReco-v1__DQMIO/397332 \
  output/dataset/raw/Muon0__Run2025E-PromptReco-v1__DQMIO/396153 \
  output/dataset/raw/Muon0__Run2025D-PromptReco-v1__DQMIO/395723 \
  output/dataset/raw/Muon0__Run2025G-PromptReco-v1__DQMIO/398902 \
  output/dataset/raw/Muon0__Run2025F-PromptReco-v1__DQMIO/396735 \
  output/dataset/raw/Muon0__Run2025G-PromptReco-v1__DQMIO/398139; do
  run=${folder##*/}
  case "$run" in
    397332) label=Good; group=good ;;
    396153|395723) label=Bad; group=bad ;;
    398902|396735|398139) label='Good test, high ae_bin score'; group=good ;;
  esac
  relative=$group/$run
  printf '## %s — Run %s\n\n| ME | LS map |\n|---|---|\n' "$label" "$run" >> "$index"
  for file in "$folder"/Muons__*.parquet; do
    uv run --locked python scripts/plot.py --input "$file"
    me=${file##*/}
    me=${me%.parquet}
    printf '| %s | [Contents](%s/%s.png) |\n' "$me" "$relative" "$me" >> "$index"
  done
  printf '\n' >> "$index"
done
