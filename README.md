# ML4DC-MuonPOG

Small LS-level Muon anomaly-detection study using cached DIALS histograms.
Eight `Muons/` MEs are used; OneHLT is excluded. OMS pile-up is a condition,
not a reconstruction target. Prescale is unused.

## Experiments

| ID | Model |
|---|---|
| `01_ae_2feat` | Preserved mean/std AE baseline: 16 → 8 → 4 → 8 → 16 |
| `02_ae_2feat_pileup` | Mean/std AE conditioned on pile-up |
| `04_mdn` | Gaussian-mixture histogram decoder conditioned on pile-up |
| `05_ae_gmm_shared` | 30 shared Gaussians per ME → 240 LS weights → conditional AE |
| `06_ae_kde` | Binned KDE inputs → conditional AE → original histogram fractions |
| `07_ae_kde_adaptive` | Training-selected spatially adaptive KDE → the same conditional AE |
| `08_ae_kde_masked` | Predict each ME from the other seven KDE histograms and pile-up |

Only the 30-component shared GMM remains. Retired independent GMM and shared
8/16/32 variants, including their results, were archived outside the repository.
The existing baseline, pile-up AE and MDN results are preserved.

For shared GMM, Gaussian positions/widths are learned once from Good training
LS, then fixed. Only LS mixture weights vary. Their total prior strength is 8.
Components describe shapes, not identified physics processes.

KDE uses bin-centre Gaussian kernels weighted by counts, with bandwidth fixed
at one native bin. It is a binned approximation, not automatic SciPy bandwidth
selection or a standalone KDE likelihood over LS features. Prior strength is
also 8; phi kernels are wrapped. Smoothing is applied to inputs only, once.
The AE reconstructs original unsmoothed bin fractions, whose scaler is fitted
on Good training targets. These targets still contain counting noise.
Both new AEs use 64/8 hidden layers and the same optimizer/epoch budget.

## Evaluation

Good 2024–2025 runs use a 70/15/15% train/validation/test split by run,
stratified by era with seed 42. Bad and Unknown runs are Test-only; all 2026
runs are Test. Saved checkpoints restore these roles during prediction.

- Select one Muon0 PromptReco processing per physical run and DCS-ready LS.
  There is no entry cut. Empty shapes are masked; missing PU is unscored.
- Good monitoring runs select epochs. Separate Good calibration runs set one
  fixed 97.5th-percentile threshold per model. Test does not select settings.
- GMM/KDE scores compare original bin fractions with predicted shapes, using
  empirical Good-training bin variance plus count-dependent sampling variance.
  Training variance includes counting noise; the score is a conservative
  residual diagnostic, not a p-value or raw likelihood.
- RR Muon labels are run-level verdicts applied to LS for reference evaluation,
  not LS-level ground truth. The current 2026 snapshot has no Muon Bad runs.
  DCS is tracker-ready selection, not Muon certification.

Test comparison, 2024–2026, snapshot 2026-11-13:

| Model | Good flagged | Good flagged, no 2026C | Bad flagged |
|---|---:|---:|---:|
| `01_ae_2feat` | 29.87% | 4.34% | 99.84% |
| `02_ae_2feat_pileup` | 30.63% | 5.36% | 100.00% |
| `04_mdn` | 26.84% | 0.36% | 51.28% |
| `05_ae_gmm_shared` | 30.94% | 5.80% | 99.84% |
| `06_ae_kde` | 28.80% | 2.95% | 92.33% |

All models score the same Test cohort: 181,011 Good LS (132,368 without
2026C) and 2,580 Bad LS. KDE reduces Good flags but misses more Bad-run LS;
it is not a replacement recommendation. PU alone does not resolve 2026C.

A marginal thinning check uses 128 high-stat Good monitoring LS, three draws
at each retained fraction. At 1% counts, GMM/KDE flag 97.92%/57.03%; extreme
low-statistics robustness remains unresolved. This cannot reproduce event-level
inter-ME correlations from cached 1D histograms. Full details are in each new
model's `metrics.json`. `06_ae_kde` remains the original fixed-bandwidth reference.

## KDE follow-up

```bash
bash runs/study_kde.sh
```

This trains only models 07/08, then evaluates five predefined variants:

- `06_ae_kde_top1`, `06_ae_kde_top2`: preserve the trained KDE AE; normalize
  ME residual scales using Good monitoring ranks, then combine the largest
  one/two terms. A rank-mean control is reported in `kde_study.json`.
- `06_ae_kde_mmd`: preserve the same AE; compare observed and predicted raw
  shapes using multiscale MMD (1/4/16 native bins), with a count-dependent
  multinomial null correction and Good-monitor rank normalization.
- `07_ae_kde_adaptive`: use spatial bandwidths of 0.5–2 native bins, with a
  per-ME multiplier selected by count-split prediction on Good TRAIN only.
  The kernels are frozen at inference; this is not LS-specific count adaptation.
- `08_ae_kde_masked`: hide one complete ME during training and fit only its
  bins. Inference makes eight held-out passes. Counts affect scoring, not the
  network input; the targets and residual formula match model 06.

The same frozen runs, seed, early-stopping budget and separate Good calibration
quantile are used. No Bad/Test results choose settings or thresholds. Comparisons
include all previously detected Bad, Regions A–D and the same marginal thinning
check. These are exploratory results on previously inspected runs, not a fresh
blind test. B is Validation and C is Train/Validation in this split: their
`regions_all_splits` results are descriptive, not held-out Test performance.
MMD corrections are diagnostics, not p-values; rank scores have
finite tail resolution. Zero scores appear only in the linear plots. No method
can recover absent low-statistics information.

Follow-up results (same snapshot; Good Test excludes 2026C in this table):

| KDE variant | Good flagged | Bad flagged | Good flagged after 1% thinning |
|---|---:|---:|---:|
| Original | 2.95% | 92.33% | 57.03% |
| Top-1 ME | 3.78% | 99.50% | 61.46% |
| Top-2 MEs | 4.62% | 99.84% | 90.63% |
| Count-corrected MMD | 3.50% | 95.00% | 0.78% |
| Adaptive KDE | 3.06% | 92.21% | 40.36% |
| Masked ME prediction | 5.12% | 92.29% | 92.97% |

There is no clear replacement. Top-2 retains all original KDE Bad flags, but
loses four LS caught by the mean/std AE while gaining four others. MMD is the
most promising low-count direction, but misses 129 previously detected Bad-run
LS; all have GlbMuon entries below 1,000 (median 58). Thinning uses 128 unique
monitoring LS and three marginal draws, not independent real-data false-positive
measurements. Adaptive/masked changes do not show a compelling gain here.
Including 2026C, Good flag rates remain 28.86–30.44%; no variant resolves that
operating-condition shift. Full cohort, per-run retention and region results
are saved in the JSON summary below.

Each variant saves its checkpoint, scores and usual plots in its own output
directory. Score-only checkpoints still identify the underlying model 06 and
store the new score transform/threshold, so `runs/predict.sh` can reuse them.
The overall summary is `output/predict/kde_study.json`. The `reconstruction/`
figures in models 06/07/08 show actual per-LS predictions for fixed Good/Bad
examples; model 06's `reference.png` only shows pooled input smoothing.

## Workflow

```bash
# Train from the existing 2024–2025 cache
bash runs/train.sh model_06_ae_kde --years 2024 2025

# Apply saved scaling and threshold to all three years
bash runs/predict.sh output/model/06_ae_kde data/dials \
  output/predict/06_ae_kde --years 2024 2025 2026 --prompt-reco

# Plot combined Test results, or inspect cached MEs
bash runs/plotting/scores.sh --input output/predict/06_ae_kde
bash runs/plotting/mes.sh
```

Use the same recipes with another model ID. Compatible checkpoints are reused;
existing predictions are protected. Use a separate `--output` for a new study.
Model commands never fetch or regenerate DIALS data. MDN training uses Apple
MPS when available; saved checkpoints support CPU prediction.

## Files and outputs

```text
scripts/
  data/                 DIALS cache and annual DCS/OMS/RR access
  models/
    config.py           paths, MEs and retained experiment IDs
    preprocessing.py    representations, PU alignment and selections
    gaussian.py         weighted Gaussian fitting and bin probabilities
    shared_gaussian.py  fixed shapes, LS weights and raw-bin residuals
    kde.py              KDE inputs and original-bin targets
    kde_scores.py       optional rank aggregation and count-corrected MMD
    networks.py         conditional AE and MDN
    core.py             calibration, scoring and checkpoint I/O
    model_0*.py         experiment definitions
  train.py              fit and calibrate
  predict.py            saved model → LS scores
  study_kde.py          fixed KDE follow-up comparison using existing data
  plotting/             ME, score, loss and confusion-matrix plots
runs/
  data/                 metadata and DIALS recipes
  train.sh              MODEL → checkpoint
  predict.sh            checkpoint + cache → predictions
  study_kde.sh          train 07/08 and run the four KDE comparisons
  plotting/             saved data/predictions → figures
data/
  dials/<dataset>/<run>/ cached ME Parquet files
  {dcs,oms,rr}/<year>.json
output/
  model/<ID>/           checkpoint, scaling and fixed threshold
  predict/<ID>/         scores.parquet and metrics.json
  plots/{mes,<ID>}/     figures
```

`Score` plots divide by the fixed threshold, so the line is 1. Test points use
Good/Bad/Unknown colors. Confusion matrices exclude Unknown and unscored LS;
`confusion_matrix_no2026C.png` uses the same cut without 2026C. `basis.png`
shows the GMM reference; KDE's `reference.png` shows raw and smoothed inputs.

OMS annual JSON keeps the existing nine LS fields. PU is joined by Run/LS and
covers all 1,276 cached PromptReco runs / 581,836 LS in the current snapshot.
Dataset folder names replace `/` with `__`; cached runs need no inventory.
Generated data/results are Git-ignored. Metadata recipes use `.venv-metadata`,
DIALS collection uses `.venv-dials`, and models use the project environment.
Credentials stay outside the repository.
