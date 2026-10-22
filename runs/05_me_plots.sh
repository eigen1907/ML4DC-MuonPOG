#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f output/model/ae_base/model.pkl ]; then
  echo "Train ae_base before plotting its features" >&2
  exit 1
fi

index=output/plots/ae_base/plots.md
rm -rf output/plots/mes output/plots/ae_base/input output/plots/ae_base/output output/plots/ae_base/diff
mkdir -p "$(dirname "$index")"
printf '# Muons ME and ae_base feature plots\n\nRaw ME plots show LS-by-bin contents. For each ae_base feature, input is the model input, output is its reconstruction, and diff is input minus output. Shape is the normalized mean bin position (with OneHLT shrinkage); stat is Good-train-standardized `log1p(entries)`.\n\n' > "$index"

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
  printf '## %s — Run %s\n\n| ME | Raw | Shape: input / output / diff | Stat: input / output / diff |\n|---|---|---|---|\n' "$label" "$run" >> "$index"
  for file in "$folder"/Muons__*.parquet; do
    if [ "$group" = good_but_anomaly ]; then
      uv run --locked python scripts/plot.py --input "$file" --good-but-anomaly
    else
      uv run --locked python scripts/plot.py --input "$file"
    fi
    me=${file##*/}
    me=${me%.parquet}
    printf '| `%s` | [ME](../mes/%s/%s.png) | [input](input/shape/%s/%s.png) / [output](output/shape/%s/%s.png) / [diff](diff/shape/%s/%s.png) | [input](input/stat/%s/%s.png) / [output](output/stat/%s/%s.png) / [diff](diff/stat/%s/%s.png) |\n' \
      "$me" "$relative" "$me" "$relative" "$me" "$relative" "$me" "$relative" "$me" \
      "$relative" "$me" "$relative" "$me" "$relative" "$me" >> "$index"
  done
  uv run --locked python scripts/plot_ae_base.py --folder "$folder" --group "$group"
  printf '\n' >> "$index"
done
