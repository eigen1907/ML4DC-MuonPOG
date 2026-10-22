#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f output/model/ae_base/model.pkl ]; then
  echo "Train ae_base before plotting standardized stat" >&2
  exit 1
fi

index=output/plots/mes/plots.md
rm -rf output/plots/mes
mkdir -p "$(dirname "$index")"
printf '# Muons ME plots\n\nEach ME has raw LS-by-bin contents (`me.png`), raw weighted mean bin position before OneHLT shrinkage (`shape.png`), and Good-train-standardized `log1p(entries)` (`stat.png`). The four `good_but_anomaly` runs sample the user-marked high-score regions; selection used offline eta statistics, while OneHLT entries can be sparse.\n\n' > "$index"

for folder in \
  output/dataset/raw/Muon0__Run2025F-PromptReco-v1__DQMIO/397332 \
  output/dataset/raw/Muon0__Run2025E-PromptReco-v1__DQMIO/396153 \
  output/dataset/raw/Muon0__Run2025D-PromptReco-v1__DQMIO/395723 \
  output/dataset/raw/Muon0__Run2025C-PromptReco-v1__DQMIO/392251 \
  output/dataset/raw/Muon0__Run2025C-PromptReco-v2__DQMIO/393240 \
  output/dataset/raw/Muon0__Run2025F-PromptReco-v1__DQMIO/396763 \
  output/dataset/raw/Muon0__Run2025G-PromptReco-v1__DQMIO/398902; do
  run=${folder##*/}
  case "$run" in
    397332) label=Good; group=good ;;
    396153|395723) label=Bad; group=bad ;;
    392251|393240|396763|398902) label='Good test, high ae_base score'; group=good_but_anomaly ;;
    *) echo "Unexpected run: $run" >&2; exit 1 ;;
  esac
  relative=$group/$run
  printf '## %s — Run %s\n\n| ME | Raw | Shape | Stat |\n|---|---|---|---|\n' "$label" "$run" >> "$index"
  for file in "$folder"/Muons__*.parquet; do
    if [ "$group" = good_but_anomaly ]; then
      uv run --locked python scripts/plot.py --input "$file" --good-but-anomaly --standardized-stat
    else
      uv run --locked python scripts/plot.py --input "$file" --standardized-stat
    fi
    me=${file##*/}
    me=${me%.parquet}
    printf '| %s | [ME](%s/%s/me.png) | [Shape](%s/%s/shape.png) | [Stat](%s/%s/stat.png) |\n' "$me" "$relative" "$me" "$relative" "$me" "$relative" "$me" >> "$index"
  done
  printf '\n' >> "$index"
done
