#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

index=output/plots/mes/plots.md
mkdir -p "$(dirname "$index")"
printf '# Raw Muons ME plots\n\n' > "$index"

for folder in \
  output/dataset/raw/Muon0__Run2025F-PromptReco-v2__DQMIO/397780 \
  output/dataset/raw/Muon0__Run2025F-PromptReco-v2__DQMIO/397765 \
  output/dataset/raw/Muon0__Run2025D-PromptReco-v1__DQMIO/395713; do
  run=${folder##*/}
  case "$run" in
    397780) label=Good ;;
    397765) label='Bad (well known)' ;;
    395713) label='Bad (ambiguous)' ;;
  esac
  relative=${folder#output/dataset/raw/}
  printf '## %s — Run %s\n\n| ME | 1D histogram | LS examples | LS map | Statistics |\n|---|---|---|---|---|\n' "$label" "$run" >> "$index"
  for file in "$folder"/Muons__*.parquet; do
    uv run --locked python scripts/plot.py --input "$file"
    me=${file##*/}
    me=${me%.parquet}
    path="$relative/$me"
    printf '| %s | [Histogram](%s/histogram.png) | [Examples](%s/examples.png) | [Contents](%s/contents.png) | [Statistics](%s/statistics.png) |\n' "$me" "$path" "$path" "$path" "$path" >> "$index"
  done
  printf '\n' >> "$index"
done
