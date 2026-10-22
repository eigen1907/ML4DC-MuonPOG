#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

index=output/plots/mes/plots.md
rm -rf output/plots/mes
mkdir -p "$(dirname "$index")"
printf '# Raw Muons ME plots\n\n' > "$index"

for folder in \
  output/dataset/raw/Muon0__Run2025F-PromptReco-v1__DQMIO/397332 \
  output/dataset/raw/Muon0__Run2025E-PromptReco-v1__DQMIO/396153 \
  output/dataset/raw/Muon0__Run2025D-PromptReco-v1__DQMIO/395723; do
  run=${folder##*/}
  case "$run" in
    397332) label=Good ;;
    396153) label='Bad (well known)' ;;
    395723) label='Bad (ambiguous)' ;;
  esac
  relative=${folder#output/dataset/raw/}
  printf '## %s — Run %s\n\n| ME | LS map |\n|---|---|\n' "$label" "$run" >> "$index"
  for file in "$folder"/Muons__*.parquet; do
    uv run --locked python scripts/plot.py --input "$file"
    me=${file##*/}
    me=${me%.parquet}
    path="$relative/$me"
    printf '| %s | [Contents](%s/contents.png) |\n' "$me" "$path" >> "$index"
  done
  printf '\n' >> "$index"
done
