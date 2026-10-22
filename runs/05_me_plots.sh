#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

index=output/plots/mes/plots.md
rm -rf output/plots/mes
mkdir -p "$(dirname "$index")"
printf '# Raw Muons ME plots\n\nThe four `good_but_anomaly` runs are high-stat examples within the four user-marked regions of the test plot. Median `GlbMuon_Glb_eta` entries/LS: 392251 (7,568), 393240 (6,136), 396763 (3,160), 398902 (3,329). The tallest plumes in the first and third regions come from low-stat runs; these are nearby high-stat cases.\n\n' > "$index"

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
