# ML4DC-MuonPOG

Small LS-level Muon DQM study using cached 2025 Muon0 histograms from DIALS.

## Five comparisons

| Directory | Input and training | LS score |
|---|---|---|
| `01_ae_2feat` | Preserved original mean/std AE, trained without DCS selection | Feature reconstruction MSE |
| `02_ae_2feat_dcs` | Same 16→8→4→8→16 AE, trained on DCS-pass Good LS | Feature reconstruction MSE |
| `03_ae_hist_dcs` | Native-histogram AE, trained on DCS-pass Good LS with MSE | Bin reconstruction MSE |
| `04_ae_dm_k1_dcs` | Same histogram architecture, trained with K=1 Dirichlet-multinomial likelihood | DM-variance-weighted reconstruction residual |
| `05_ae_dm_k1_weighted_window` | Reuses the stage-04 checkpoint; no additional training | Centered five-LS mean of the stage-04 score |

**There is no phi-entry or minimum-statistics cut.** Stages 2–4 use all DCS-pass LS from Good training runs, including sparse LS. The old entry-cut fits are not reused. Stage 1 is the original checkpoint, preserved byte-for-byte. Stage 5 stores a small source reference instead of duplicating stage 4's weights.

On a fresh checkout without the stage-01 checkpoint, the runner can refit its historical unfiltered recipe and marks that origin in the pickle. This local study reused the original; it did not refit stage 01.

Eight Muons MEs are used: global-muon eta/phi/pT and isolation, HM/LM dimuon masses, global event-average hits, and standalone-track hits. OneHLT is excluded from training. Mean/std gives 16 features; the histogram models use 980 normalized native bins, eight training-standardized log1p(N) values, and eight nonempty masks, through a 996→64→4→64 network with histogram heads. Eta is folded only in the historical moment features.

K=1 means one DM distribution per ME. Its concentration is learned per ME and shared across LS. N controls sampling uncertainty; there is no separate rate/efficiency prediction. Stage 4 already uses variance weighting; stage 5 adds only the LS window. The count-rank, CDF, masked-ME, denoising, tree, covariance and count-model branches are no longer active.

## Run

```bash
uv sync
bash runs/03_study.sh               # fit/resume stages 2–4, score all five
bash runs/03_study.sh --plots-only  # redraw saved scores
bash runs/04_me_plots.sh            # normalized raw MEs and entries plots
uv run --locked python -m unittest discover -s tests
```

The study reads the existing Parquet cache and never calls DIALS. Acquisition is separate: `01_example.sh` retrieves one ME/run; `02_all_2025.sh` resumes the 2025 Muon0 collection. Both use existing CERN credentials and the `muo` workspace.

`03_study.sh` records progress in `output/logs/study.log` and its exit code in `study.status`. It uses `caffeinate` to prevent idle sleep on this Mac; keep the lid open. To detach from a terminal:

```bash
zellij attach --create-background muon-ae-study -- bash "$PWD/runs/03_study.sh"
tail -f output/logs/study.log
```

## Data and evaluation

- Cached catalogue: 744 PromptReco runs / 301,652 LS; 727 provisional Good, eight Known Bad and nine Ambiguous Bad run labels.
- Historical era-stratified Good run split: 70% train, 15% validation, 15% test, seed 42. Both Bad categories are test-only.
- Stages 2–4 train on **DCS-pass Good training LS only**, with no entry cut. The public DCS snapshot is tracker-ready selection, not a Muon quality label or Golden JSON.
- Validation runs have disjoint monitoring and calibration roles. Up to 4,096 DCS-pass monitoring LS select histogram checkpoints. All finite Good calibration scores set each stage's fixed 97.5th-percentile threshold.
- Special high-pileup runs 398902/398903 do not define normal training/calibration references. D=398902 remains an evaluation example.
- Histogram N=0 targets provide no shape evidence and are excluded from loss/score. An all-empty histogram LS remains unscored. Historical feature AEs retain their original zero-valued empty moments.
- Stage 5 uses up to two later LS as well as two earlier LS. It resets at dataset/run boundaries, LS gaps and DCS-state changes. This is an offline contextual score and can dilute a brief fault; stage 4's instantaneous result is retained.
- Thresholds are calibrated separately after each score transformation. Scores are not p-values or P(bad). Reused Test results and provisional run labels do not establish LS-level false/true-positive rates.

`data/metadata/known_bad_reference.csv` freezes the **507 previously detected Known Bad LS**. The audit reports all recorded LS and the 504 DCS-pass LS separately, including newly lost or unscored detections. Bad labels never fit models or thresholds.

The completed DCS-only fit uses **198,387 Good training LS** (the frozen stage-01 checkpoint used 211,603 unfiltered LS). Current descriptive Test results:

| Stage | Known Bad flagged / 504 DCS-pass LS | Provisional Good flagged |
|---|---:|---:|
| 01 | 503 | 8.35% |
| 02 | **504** | 7.89% |
| 03 | 154 | 7.01% |
| 04 | 502 | 3.56% |
| 05 | **504** | 4.36% |

Stages 02 and 05 also retain all **507 recorded Known Bad LS** before DCS gating. Stage 04 loses two previous detections; stage 03 remains a negative control, not a certification candidate. Histogram early stopping selected epoch 19/27 for stage 03 and 40/48 for stage 04. These reused cases are regression checks, not independent validation or a guarantee for future faults.

## Files and output

| File | Role |
|---|---|
| `scripts/config.py` | Shared paths, ME schema, experiment names, seed and fixed review examples |
| `scripts/data.py` | Cache, native counts/ROOT entries, moments, DCS and fixed run split |
| `scripts/models.py` | Feature AE and MSE/DM histogram AEs |
| `scripts/scoring.py` | Residuals, calibration, LS windows and Known Bad retention |
| `scripts/study.py` | One explicit runner for the five comparisons |
| `scripts/plots.py` | Shared CMS-style score/loss/comparison plots |
| `scripts/diagnostics.py` | Flow coverage and Run 393240 count/shape evolution |
| `scripts/plot.py` | Raw normalized ME heatmaps and entries distributions |
| `scripts/fetch.py`, `collect.py` | Separate DIALS acquisition |

The runner calls data loading, model fitting, scoring and plotting directly. ME/feature dimensions follow the selected ME list rather than fixed array sizes; diagnostics look up MEs by name. Edit shared study choices in `config.py`, training settings in `models.py`, and the calibration quantile in `scoring.py`. A–D and representative run numbers are explicit review examples, not model selections.

Existing checkpoints are reused only when the data signature agrees. Histogram checkpoints also check ME order, physical binning and training scaling. A changed selection requires an explicit refit; a routine rerun preserves the checkpoint files. Tests cover sparse/empty LS, run separation, calibration isolation, checkpoint resume and unchanged scores.

```text
data/run_labels.csv              provisional run labels
data/metadata/                   DCS snapshot, provenance and frozen detection reference
output/dataset/                  unchanged raw Parquet cache and run catalogue
output/model/{01..05}/           four checkpoints; stage 05 source.json
output/predict/{01..05}/         LS scores, ME terms, thresholds and protocol signatures
output/predict/comparison/       cohort summary and full Known Bad detection audit
output/plots/{01..05}/           test, distribution and applicable loss plots
output/plots/comparison/         aligned five-stage and Known Bad views
output/plots/mes/                representative raw MEs
output/{predict,plots}/diagnostics/  flow coverage and Region B trends
```

Plots use consecutive recorded LS positions with run-number ticks; the B diagnostic uses actual lumisection numbers. Each score is displayed divided by its own threshold. Linear views show 0–5 and `_log` views use a log scale. Fonts use the CMS style defaults and figures have no titles.

ROOT fills and in-range bin totals remain separate: only about 23% of GlobalHits fills are within its stored range in Good training data. This is not a detector efficiency. Region B (Run 393240) contains both declining counts and changing shapes; its quality remains unresolved. Normal operating-state changes should be studied with independent conditions, without fitting this Test run away.

Earlier studies are preserved outside the active repository in `../.codex-backups/ML4DC-MuonPOG-20261112-113702/`, including a source snapshot and old checkpoints/predictions/plots. A compact historical comparison and the old B-thinning/flow diagnostics remain under `output/predict/diagnostics/previous_*`; they used the previous entry-cut fits and must not be confused with the new DCS-only results. Generated cache, checkpoints and plots are ignored by Git; code, labels, DCS provenance and `uv.lock` are versioned.
