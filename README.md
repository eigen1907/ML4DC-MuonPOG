# ML4DC-MuonPOG

Small LS-level Muon anomaly-detection study using cached DIALS histograms.
Eight `Muons/` MEs are used; OneHLT is excluded. New models use OMS pile-up
as a condition, not a reconstruction target. Prescale is not a model input.

## Experiments

| ID | Model |
|---|---|
| `01_ae_2feat` | Preserved mean/std AE baseline: 16 → 8 → 4 → 8 → 16 |
| `02_ae_2feat_pileup` | Mean/std AE conditioned on pile-up; reconstructs only ME features |
| `03_ae_gmm` | Fit each LS/ME histogram with Gaussians, then reconstruct the fitted shape parameters with a pile-up-conditioned AE |
| `04_mdn` | Pile-up-conditioned Gaussian-mixture histogram density model |
| `05_ae_gmm_shared` | Fixed Gaussian shapes learned from Good training LS; LS-specific weights reconstructed with a pile-up-conditioned AE |

GMM fits with 2, 4 and 8 components are compared. The retained settings and
network definitions live in `scripts/models/model_0*.py`. Gaussian components
describe histogram shapes; they are not labels for individual physics processes.
GMM uses bin-centre EM; low-stat component parameters can be unstable. MDN
integrates Gaussian shapes over native bins, with a linear approximation for
phi. Its deviance score is a diagnostic, not a calibrated probability.
MDN receives histogram fractions, log-counts, observation masks and pile-up;
it predicts shapes rather than counts. Counts enter the deviance score.

The retained GMM uses K=8, a 64/8 AE bottleneck and a 20-iteration EM cap;
80 iterations improved the raw fits but worsened held-out AE reconstruction.
The retained MDN uses K=8 and a 64/8 encoder. Candidates were selected on Good
monitoring runs: four moment-AE settings, six GMM settings plus EM refinement,
and six MDN settings with the best two refitted on all training LS.

Model 05 is one controlled small study: K=8 shared means/widths, 64 weight
features, the same 64/8 AE bottleneck as model 03, and total prior strength 8.
The primary score compares original bin fractions with reconstructed shapes,
using training bin variance plus count-dependent sampling variance. Training
variance includes counting noise; this conservative diagnostic is not a p-value.
Its score also differs from model 03, so results do not isolate representation alone.

## Current result

Fixed-threshold Test comparison, 2024–2026 (2026-11-13 snapshot):

| Model | Good flagged | Good flagged, no 2026C | Bad flagged |
|---|---:|---:|---:|
| `01_ae_2feat` | 29.87% | 4.34% | 99.84% |
| `02_ae_2feat_pileup` | 30.63% | 5.36% | 100.00% |
| `03_ae_gmm` | 30.30% | 5.22% | 100.00% |
| `04_mdn` | 26.84% | 0.36% | 51.28% |
| `05_ae_gmm_shared` | 31.96% | 7.24% | 99.15% |

These are LS in RR-labelled runs: 181,011 Good LS (132,368 without 2026C)
and 2,580 Bad LS. All five models score the same Test cohort. Baseline remains the
reference: PU/GMM did not improve this trade-off, while MDN reduces Good flags
but misses many Bad-run LS. PU alone does not resolve 2026C.
A matched no-PU version of model 02 gave 5.19% Good flagged without 2026C and
100% Bad flagged, so a PU benefit was not demonstrated. Only one representative
per family is retained; other candidates and retired DM/window results are removed
from the active project. Existing results were backed up outside the repository.

For model 05, auxiliary **weight-reconstruction MSE**, with its own unchanged
Good-calibration protocol, gives 30.14% Good flagged overall, 4.69% without
2026C, and 100% Bad flagged. This is a diagnostic; it did not replace the primary
score after looking at Test. Shared weights modestly improve on model 03's
5.22%/100% trade-off, while the original-bin score still needs work.

Marginal thinning of 128 high-stat Good monitoring LS (three draws each):

| Counts retained | Model 03 flagged | Model 05 primary flagged |
|---|---:|---:|
| 100% | 0% | 0% |
| 10% | 7.55% | 32.29% |
| 1% | 100% | 72.14% |

At 1%, median predicted-shape change (total variation) falls from 0.071 to
0.022, with 99.48% of MEs still observed. Fitting is more stable, but anomaly
robustness is not uniformly improved. Cached 1D histograms cannot reproduce
event-level inter-ME correlations. Model 05's `basis.png` shows the fixed
components and pooled Good-training shape; `metrics.json` contains the study.

## Workflow

```text
Cached MEs + OMS pile-up → preprocessing → train → checkpoint → predict → plot
```

```bash
# Fit using 2024–2025 only
bash runs/train.sh model_02_ae_2feat_pileup --years 2024 2025

# Predict all three years using the saved model, scaling and fixed threshold
bash runs/predict.sh output/model/02_ae_2feat_pileup data/dials \
  output/predict/02_ae_2feat_pileup --years 2024 2025 2026 --prompt-reco

# Draw the combined Test; no fitting or recalibration
bash runs/plotting/scores.sh --input output/predict/02_ae_2feat_pileup

# Inspect normalized cached MEs
bash runs/plotting/mes.sh
```

Use the same recipes with another model ID. Existing predictions are protected;
use a new `--output` directory for a separate experiment. No model command fetches
or regenerates DIALS data.
MDN training uses Apple MPS when available; saved checkpoints support CPU prediction.

## Data and evaluation

- **Split:** Good runs from 2024–2025 are split 70/15/15% by run, stratified by era
  with seed 42. Bad and Unknown runs are Test-only; all 2026 runs are also Test.
  Prediction restores the saved run roles.
- **Selection:** one Muon0 PromptReco processing per physical run, with DCS-ready
  LS. There is no entry cut. Empty histogram shapes are masked where required;
  missing pile-up is never replaced with zero.
- **Validation:** monitoring runs select settings and epochs. Separate Good
  calibration runs set the 97.5th-percentile threshold. Test does not select
  settings, scalers or thresholds.
- **OMS:** `data/oms/<year>.json` holds the existing nine LS fields, including
  pile-up. The 2026-11-13 snapshot covers all 1,276 cached PromptReco runs and
  581,836 DIALS LS. Pile-up is joined by Run/LS; prescale is unused.
- **Labels:** annual RR Muon verdicts give Good/Bad/Unknown run labels. These are
  applied to LS for reference evaluation, not LS-level ground truth. The current
  2026 RR snapshot has no Muon Bad runs. DCS is a tracker-ready selection, not
  Muon certification.

Model plots combine the 2024–2026 Test. `Score` is anomaly score divided by the
fixed threshold, so the threshold line is 1. Points use Good/Bad/Unknown colors;
LS are consecutive and sparse x ticks show Run numbers. Confusion matrices show
LS counts and row percentages, excluding Unknown and unscored LS.
`confusion_matrix_no2026C.png` repeats the evaluation without 2026C, at the same
threshold. It is a diagnostic alongside the full Test.

## Structure

```text
scripts/
  data/                  DIALS cache and annual DCS/OMS/RR access
  models/
    config.py            paths, selected MEs and experiment IDs
    preprocessing.py     ME features, pile-up alignment and selections
    gaussian.py          binned Gaussian-mixture histogram fitting
    shared_gaussian.py   fixed shapes, LS weights and original-bin residuals
    networks.py          conditional AE and histogram mixture networks
    core.py              shared scoring, calibration and checkpoint I/O
    model_0*.py          retained experiment definitions
  train.py               fit and calibrate one model
  predict.py             apply a saved checkpoint
  plotting/              ME, score, loss and confusion-matrix plots
runs/
  data/                  metadata and DIALS recipes
  train.sh               MODEL → checkpoint
  predict.sh             checkpoint + cache → predictions
  plotting/              saved predictions or MEs → figures
data/
  dials/<dataset>/<run>/  cached ME Parquet files
  {dcs,oms,rr}/<year>.json
output/
  model/<ID>/            checkpoint, fitted scaling and threshold
  predict/<ID>/          scores.parquet and metrics.json
  plots/{mes,<ID>}/      ME figures and combined Test results
```

Dataset directory names replace `/` with `__`; cached runs need no inventory.
Generated data and results are ignored by Git. Metadata recipes use
`.venv-metadata`, DIALS collection uses `.venv-dials`, and model recipes use the
project environment. Credentials stay outside the repository.
