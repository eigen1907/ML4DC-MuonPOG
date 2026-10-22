#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f output/model/ae_base/model.pkl ]; then
  echo "Train ae_base before plotting its features" >&2
  exit 1
fi

index=output/plots/ae_base/plots.md
rm -rf output/plots/mes output/plots/ae_base/{input,output,diff,stat,shape}
mkdir -p "$(dirname "$index")"
cat > "$index" <<'EOF'
# Muons ME and ae_base feature plots

The four Good-test clumps on [test.png](test.png) are:

- A (2025B, 610 LS): 391550, 391670, 391675, 391677, 391682, 391826, 391906, 391953, 392088, 392090
- B (2025C, 4,777 LS): 393240
- C (2025F, 1,507 LS): 396727, 396735, 396757, 396763
- D (2025G, 428 LS): 398902

Each region figure concatenates all LS in run order, with run numbers on the x axis. Below-threshold and low/zero-stat LS remain. Raw ME plots show LS-by-bin contents. Each feature plot overlays input and AE output above their difference (input minus output). Shape is the normalized mean bin position (with OneHLT shrinkage); stat is Good-train-standardized `log1p(entries)`.

EOF

for selection in 397332 396153 395723 A B C D; do
  case "$selection" in
    397332)
      label='Good — Run 397332'; group=good; relative=good/397332
      folders=(output/dataset/raw/Muon0__Run2025F-PromptReco-v1__DQMIO/397332) ;;
    396153)
      label='Bad — Run 396153'; group=bad; relative=bad/396153
      folders=(output/dataset/raw/Muon0__Run2025E-PromptReco-v1__DQMIO/396153) ;;
    395723)
      label='Bad — Run 395723'; group=bad; relative=bad/395723
      folders=(output/dataset/raw/Muon0__Run2025D-PromptReco-v1__DQMIO/395723) ;;
    A)
      label='Good test — Region A'; group=A; relative=good_but_anomaly/A
      folders=(output/dataset/raw/Muon0__Run2025B-PromptReco-v1__DQMIO/{391550,391670,391675,391677,391682,391826,391906,391953,392088,392090}) ;;
    B)
      label='Good test — Region B'; group=B; relative=good_but_anomaly/B
      folders=(output/dataset/raw/Muon0__Run2025C-PromptReco-v2__DQMIO/393240) ;;
    C)
      label='Good test — Region C'; group=C; relative=good_but_anomaly/C
      folders=(output/dataset/raw/Muon0__Run2025F-PromptReco-v1__DQMIO/{396727,396735,396757,396763}) ;;
    D)
      label='Good test — Region D'; group=D; relative=good_but_anomaly/D
      folders=(output/dataset/raw/Muon0__Run2025G-PromptReco-v1__DQMIO/398902) ;;
  esac
  files=()
  for folder in "${folders[@]}"; do
    files+=("$folder"/Muons__*.parquet)
  done
  printf '## %s\n\n| ME | Raw | Stat | Shape |\n|---|---|---|---|\n' "$label" >> "$index"
  if [ "$group" = good ] || [ "$group" = bad ]; then
    uv run --locked python scripts/plot.py --input "${files[@]}"
  else
    uv run --locked python scripts/plot.py --input "${files[@]}" --region "$group"
  fi
  for file in "${folders[0]}"/Muons__*.parquet; do
    me=${file##*/}
    me=${me%.parquet}
    printf '| `%s` | [ME](../mes/%s/%s.png) | [Stat](stat/%s/%s.png) | [Shape](shape/%s/%s.png) |\n' \
      "$me" "$relative" "$me" "$relative" "$me" "$relative" "$me" >> "$index"
  done
  uv run --locked python scripts/plot_ae_base.py --folder "${folders[@]}" --group "$group"
  printf '\n' >> "$index"
done
