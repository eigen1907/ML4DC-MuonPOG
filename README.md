# ML4DC-MuonPOG

Small LS-level Muon anomaly-detection study using cached DIALS histograms.
Core models use eight `Muons/` MEs; OneHLT is excluded. Models 07/08 use eta
only. OMS pile-up is a condition, not a reconstruction target. Prescale is unused.

## Models

| ID | Model |
|---|---|
| `01_ae_2feat` | Preserved mean/std AE baseline: 16 → 8 → 4 → 8 → 16 |
| `02_ae_2feat_pileup` | Mean/std AE conditioned on pile-up |
| `04_mdn` | Gaussian-mixture histogram decoder conditioned on pile-up |
| `05_ae_gmm_shared` | 30 shared Gaussians per ME → 240 LS weights → conditional AE |
| `06_ae_kde` | Binned KDE inputs → conditional AE → original histogram fractions |
| `06_ae_kde_mmd` | Reuse the KDE AE, score shapes with count-corrected MMD |
| `07_eta_betabinomial` | Statistical eta acceptance-loss test; no neural network |
| `08_eta_fault_classifier` | Eta MLP classifier trained on normal/synthetic-fault pairs |

Models 01–06 use `runs/train.sh` and `runs/predict.sh`. Models 07/08 are
standalone eta-only prototypes run together with `bash runs/compare_eta.sh`;
they are not registered with the core train/predict CLI. Neither is an autoencoder.

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
The GMM and KDE AEs use 64/8 hidden layers. MMD reuses the KDE weights and
training history; it changes scoring and calibration only.

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
| `06_ae_kde_mmd` | 29.14% | 3.50% | 95.00% |

All models score the same Test cohort: 181,011 Good LS (132,368 without
2026C) and 2,580 Bad LS. KDE/MMD reduce Good flags but miss more Bad-run LS;
neither is a replacement recommendation. PU alone does not resolve 2026C.

A marginal thinning check uses 128 high-stat Good monitoring LS, three draws
at each retained fraction. At 1% counts, GMM/KDE/MMD flag 97.92%/57.03%/0.78%.
MMD improves this stress check, with a Bad-sensitivity trade-off. This cannot
reproduce event-level inter-ME correlations from cached 1D histograms. Details
are in each model's `metrics.json`. `06_ae_kde` remains the fixed-bandwidth reference.

## KDE scoring

`06_ae_kde` uses the original variance-weighted bin residuals.
`06_ae_kde_mmd` compares raw observed and predicted shapes using Gaussian MMD
kernels of 1/4/16 native bins. It corrects for expected multinomial counting
fluctuations, normalizes each ME using Good-monitor ranks, then averages the
observed ME scores. Separate Good calibration LS set the fixed threshold.
These are diagnostic scores, not p-values; zero scores appear only in linear plots.

MMD misses 129 Bad-run LS caught by the mean/std AE; their GlbMuon in-range
counts are 1–225 (median 58). Some have substantial per-ME evidence diluted by
the final average: Run 395719 / LS95 has eta MMD 62.77, yet final score 1.598 is
below the 1.703 cutoff. Rank scores also have a finite tail ceiling. Other LS
have genuinely little shape information, such as a single observed muon.
A Bad run label alone does not establish an observable defect in every LS.

Only the original KDE and MMD variant remain active. The KDE `reference.png`
shows pooled input smoothing; its `reconstruction/` figures show actual LS
predictions. MMD's `metrics.json` includes full-cohort results, thinning checks
and descriptive Regions A–D diagnostics. B/C include
training/validation runs, so their diagnostics are not independent Test results.

## Top-N MMD comparison

```bash
bash runs/compare_mmd.sh
```

This compares the largest 1/2/3 normalized ME scores **within each LS**.
The selected MEs may differ between LS. Observed zeros remain observations;
empty MEs are missing, and fewer than N observed MEs use those available.
AE weights, count corrections and Good-monitor ECDFs stay fixed. Each N gets
its own 97.5th-percentile threshold from the same separate Good calibration LS.

| MMD aggregation | Good flagged, no 2026C | Bad flagged | Previously flagged MMD Bad lost |
|---|---:|---:|---:|
| All observed MEs | 3.50% | 95.00% | 0 |
| Top-1 | 4.15% | 95.66% | 15 |
| Top-2 | 4.24% | 95.54% | 11 |
| Top-3 | 3.94% | 95.47% | 9 |

All three recover Run 395719 / LS95, but overall gains are small and some old
Bad detections are lost. At 1% marginal thinning, Good flag fractions are
0.78% (mean), 3.13% (Top-1), 1.82% (Top-2) and 1.30% (Top-3).
Top-1 calibration flags 2.455% due to threshold ties; Top-2/3 flag 2.500%.
This previously inspected Test is exploratory, so it does not select a new
default N. The original mean checkpoint and results remain unchanged.

Checkpoints, scores and plots are nested under each area's
`06_ae_kde_mmd/top1/`, `top2/`, `top3/`. The summary is
`output/predict/06_ae_kde_mmd/topn.json`; the shared AE loss stays in the parent
plot folder. No extra model definitions or training jobs are needed.

## 07 / 08: eta models

```bash
bash runs/compare_eta.sh
```

These two isolated prototypes use only the native signed eta histogram, its
in-range count and saved PU. Both use the same 16,384 Good TRAIN parents;
4,096 Good monitoring LS and separate Good calibration runs keep their roles.
The calibration has 42,192 usable eta LS (six empty observations are masked).

- **07 — Beta-binomial eta-loss model:** a statistical count model, not a
  neural network. Four TRAIN-PU strata learn regional beta-binomial references.
  Central `|eta| < 0.9` and forward `|eta| > 1.2` loss hypotheses produce a
  profile likelihood-ratio score. Fitted loss is a reference-relative shape
  parameter, not a measured detector efficiency or a p-value.
- **08 — Synthetic-fault MLP classifier:** native fractions + log1p(N) + PU,
  `102 → 64 → 16 → 1`. It predicts a fault score, not a reconstructed histogram.
  Good-derived normal/fault pairs have identical N and PU. Central/forward
  losses of 25–95% are simulated; fixed monitoring-pair BCE selects the epoch.
  Softplus(logit) preserves score tails; it is not a Bad probability.

| Prototype | Good flagged, no 2026C | Bad flagged | MMD misses recovered | 1% thinning Good flagged |
|---|---:|---:|---:|---:|
| `07_eta_betabinomial` | 2.51% | 2579/2580 | 128/129 | 3.39% |
| `08_eta_fault_classifier` | 3.95% | 2580/2580 | 129/129 | 61.72% |

The hypothesis prototype retains every MMD Bad flag; its one miss is
Run 396153 / LS1 with one observed muon. The classifier also flags uncertain
low-count normal samples, so its perfect Bad coverage is not a robust win.
Both leave 38 empty-eta Good Test LS unscored, explicitly counted in metrics.
Full-Test Good flags remain 28.48%/29.35%, so 2026C is still unresolved.
The hypothesis has little sensitivity to unseen left/right eta losses;
neither prototype is a general replacement for the eight-ME models.

Synthetic diagnostics use 64 held-out Good-Test 2024/25 templates at fixed N
and severity, including unseen left/right losses. They are simulations, not
independent detector-fault truth. These Test runs were previously inspected;
no settings were selected from the new Bad results. Existing experiments and
production defaults are unchanged.

Each model now has its own directory alongside the other models:

```text
output/model/07_eta_betabinomial/model.json
output/model/08_eta_fault_classifier/model.pt
output/predict/{07_eta_betabinomial,08_eta_fault_classifier}/
output/plots/{07_eta_betabinomial,08_eta_fault_classifier}/
```

The comparison summary is `output/predict/eta_comparison.json`; the shared
synthetic sensitivity figure is `output/plots/eta_power.png`. Only model 08
has a training loss curve. Both use the dedicated `compare_eta.sh` recipe;
existing core models and defaults are unchanged.

## Workflow

```bash
# Train from the existing 2024–2025 cache
bash runs/train.sh model_06_ae_kde --years 2024 2025

# Reuse the KDE weights; fit MMD normalization/threshold only
bash runs/train.sh model_06_ae_kde_mmd --years 2024 2025

# Apply the saved model and threshold to all three years
bash runs/predict.sh output/model/06_ae_kde_mmd data/dials \
  output/predict/06_ae_kde_mmd --years 2024 2025 2026 --prompt-reco

# Plot combined Test results, or inspect cached MEs
bash runs/plotting/scores.sh --input output/predict/06_ae_kde_mmd
bash runs/plotting/mes.sh
```

Use the same recipes with another core model ID (01–06). Compatible checkpoints are reused;
existing predictions are protected. Use a separate `--output` for a new study.
Model commands never fetch or regenerate DIALS data. MDN training uses Apple
MPS when available; saved checkpoints support CPU prediction.

## Files and outputs

```text
scripts/
  data/                 DIALS cache and annual DCS/OMS/RR access
  models/
    config.py           paths, MEs and core train/predict model IDs
    preprocessing.py    representations, PU alignment and selections
    gaussian.py         weighted Gaussian fitting and bin probabilities
    shared_gaussian.py  fixed shapes, LS weights and raw-bin residuals
    kde.py              KDE inputs and original-bin targets
    kde_scores.py       count-corrected MMD and Good-monitor ranks
    networks.py         conditional AE and MDN
    core.py             calibration, scoring and checkpoint I/O
    model_0*.py         numbered per-model implementations
  train.py              fit and calibrate
  predict.py            saved model → LS scores
  compare_mmd.py        frozen MMD top-1/2/3 scoring comparison
  compare_eta.py        compare model 07 (statistics) and 08 (MLP)
  plotting/             ME, score, loss and confusion-matrix plots
runs/
  data/                 metadata and DIALS recipes
  train.sh              MODEL → checkpoint
  predict.sh            checkpoint + cache → predictions
  compare_mmd.sh        rescore cached MMD terms and check low-statistics robustness
  compare_eta.sh        run models 07/08 with the same frozen eta data
  plotting/             saved data/predictions → figures
data/
  dials/<dataset>/<run>/ cached ME Parquet files
  {dcs,oms,rr}/<year>.json
output/
  model/<ID>/           fitted parameters/checkpoint and fixed threshold
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
