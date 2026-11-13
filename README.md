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
model's `metrics.json`. KDE is one fixed-bandwidth first test, not an optimized
KDE architecture study.

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
    networks.py         conditional AE and MDN
    core.py             calibration, scoring and checkpoint I/O
    model_0*.py         experiment definitions
  train.py              fit and calibrate
  predict.py            saved model → LS scores
  plotting/             ME, score, loss and confusion-matrix plots
runs/
  data/                 metadata and DIALS recipes
  train.sh              MODEL → checkpoint
  predict.sh            checkpoint + cache → predictions
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
