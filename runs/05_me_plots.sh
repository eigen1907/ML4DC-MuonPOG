#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

index=output/plots/mes/plots.md
rm -rf output/plots/mes
mkdir -p "$(dirname "$index")"
printf '# Raw Muons ME plots\n\nThe four `good_but_anomaly` runs correspond to high-score regions in `ae_base/test.png`. The rise near the 394000 tick belongs to run 393240 because LS are concatenated on the horizontal axis.\n\n' > "$index"

for folder in \
  output/dataset/raw/Muon0__Run2025F-PromptReco-v1__DQMIO/397332 \
  output/dataset/raw/Muon0__Run2025E-PromptReco-v1__DQMIO/396153 \
  output/dataset/raw/Muon0__Run2025D-PromptReco-v1__DQMIO/395723 \
  output/dataset/raw/Muon0__Run2025B-PromptReco-v1__DQMIO/391670 \
  output/dataset/raw/Muon0__Run2025C-PromptReco-v2__DQMIO/393240 \
  output/dataset/raw/Muon0__Run2025F-PromptReco-v1__DQMIO/396727 \
  output/dataset/raw/Muon0__Run2025G-PromptReco-v1__DQMIO/398902; do
  run=${folder##*/}
  case "$run" in
    397332) label=Good; group=good ;;
    396153|395723) label=Bad; group=bad ;;
    391670|393240|396727|398902) label='Good test, high ae_base score'; group=good_but_anomaly ;;
    *) echo "Unexpected run: $run" >&2; exit 1 ;;
  esac
  relative=$group/$run
  printf '## %s — Run %s\n\n| ME | LS map |\n|---|---|\n' "$label" "$run" >> "$index"
  for file in "$folder"/Muons__*.parquet; do
    if [ "$group" = good_but_anomaly ]; then
      uv run --locked python scripts/plot.py --input "$file" --good-but-anomaly
    else
      uv run --locked python scripts/plot.py --input "$file"
    fi
    me=${file##*/}
    me=${me%.parquet}
    printf '| %s | [Contents](%s/%s.png) |\n' "$me" "$relative" "$me" >> "$index"
  done
  printf '\n' >> "$index"
done
