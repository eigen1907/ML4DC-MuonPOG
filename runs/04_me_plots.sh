#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -f output/model/ae_base/model.pkl ]; then
  echo "Train ae_base before plotting its features" >&2
  exit 1
fi

index=output/plots/ae_base/plots.md
rm -rf output/plots/mes output/plots/ae_base/{stat,mean,std}
mkdir -p "$(dirname "$index")"
cat > "$index" <<'EOF'
# Muons ME and ae_base feature plots

The four regions selected from the previous mean-only logarithmic test view are marked on [test_log.png](test_log.png):

- A (2025B, 610 LS): 391550, 391670, 391675, 391677, 391682, 391826, 391906, 391953, 392088, 392090
- B (2025C, 4,777 LS): 393240
- C (2025F, 1,507 LS): 396727, 396735, 396757, 396763
- D (2025G, 428 LS): 398902

ME heatmaps concatenate recorded LS with positive stored-bin sums, in run/LS order, without gaps. The Lumisection x axis is a consecutive 1-based nonempty-LS index, not an actual LS ID; ME-specific filtering means indices differ across MEs. Multi-run boundaries follow the retained rows. Each LS is normalized as p_i = n_i / sum(n_i) over all stored in-range bins before display zoom. Entirely empty MEs keep their plot paths with a "No nonempty LS" placeholder. The ME name appears in white at the upper left. This display filtering leaves data and the model unchanged. All AE mean/std/stat plots retain all recorded LS, including below-threshold and low/zero-stat LS, in run/LS order. Their Lumisection axis is a consecutive 1-based recorded-LS index, not the original LS ID; multi-run boundaries remain marked, without run-number ticks. Mean/std plots overlay input and AE output above their difference (input minus output). Mean and std are calculated directly from histogram bin positions normalized to 0–1, without low-stat correction. Empty histograms use mean = std = 0 when no distribution is observed. The two MuonRecoOneHLT MEs are excluded from AE inputs; their normalized ME and entries plots remain available. Entries plots show raw statistics for diagnosis only; entries are not AE inputs or part of the anomaly score.

Each case also has `entries.png`: six colored step histograms overlaid on shared geometric bins with a separate zero bin. Entries use a symlog x axis (linear threshold 1, upper limit 10^6); LS counts use a linear y axis. All recorded LS, including zero-entry LS, contribute. Legend labels are GlbMuon, GlbMuonOneHLT, GlbGlbMuonHM, GlbGlbMuonLM, GlbMuonHits, and StaMuonHits, each followed by `μ = x,xxx.x`, the arithmetic entries/LS mean including zeros. Exact per-(run, LS) entries vectors are deduplicated: Analyzer eta/phi/pt and Isolation share one histogram; OneHLT eta/pt share another. Entries count ME fills/objects; they need not equal event counts or the sum of stored bins. These diagnostic plots leave the model unchanged.

Diff panels use fixed linear ranges: mean ±0.1 and std ±0.02. Up/down triangles indicate residuals beyond the displayed range; the upper panel shows the full input and output values.

EOF

for selection in 397332 396153 395723 397765 A B C D; do
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
    397765)
      label='Bad — Run 397765'; group=bad; relative=bad/397765
      folders=(output/dataset/raw/Muon0__Run2025F-PromptReco-v2__DQMIO/397765) ;;
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
  printf '## %s\n\n[Entries distribution](../mes/%s/entries.png)\n\n| ME | Normalized ME | Entries (diagnostic) | Mean | Std |\n|---|---|---|---|---|\n' "$label" "$relative" >> "$index"
  if [ "$group" = good ] || [ "$group" = bad ]; then
    uv run --locked python scripts/plot.py --input "${files[@]}"
  else
    uv run --locked python scripts/plot.py --input "${files[@]}" --region "$group"
  fi
  for file in "${folders[0]}"/Muons__*.parquet; do
    me=${file##*/}
    me=${me%.parquet}
    if [[ "$me" == Muons__MuonRecoOneHLT__* ]]; then
      printf '| `%s` | [ME](../mes/%s/%s.png) | [Entries](stat/%s/%s.png) | — (not an AE input) | — (not an AE input) |\n' \
        "$me" "$relative" "$me" "$relative" "$me" >> "$index"
    else
      printf '| `%s` | [ME](../mes/%s/%s.png) | [Entries](stat/%s/%s.png) | [Mean](mean/%s/%s.png) | [Std](std/%s/%s.png) |\n' \
        "$me" "$relative" "$me" "$relative" "$me" "$relative" "$me" "$relative" "$me" >> "$index"
    fi
  done
  uv run --locked python scripts/plot_ae_base.py --folder "${folders[@]}" --group "$group"
  printf '\n' >> "$index"
done
