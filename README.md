# ML4DC-MuonPOG

LS-level Muon reconstruction anomaly detection using eight cached Muon0 DQM
histograms from 2024–2026. The main experiment is
**`ls_group_1pb / 03_ae_gmm_deviance`**. Other models and LS modes remain as comparisons.

## Run

```bash
uv sync --locked
bash runs/gmm.sh                    # Main: 1 pb^-1 groups, 03 GMM deviance
bash runs/plotting/good_runs.sh      # Reproduce the fixed Good-run investigation
```

The main recipe trains, predicts, evaluates and draws Gaussian examples.
Compatible checkpoints, predictions and Prefit caches are reused; no DIALS
requests are made. To redraw saved results without training:

```bash
uv run --locked python -m scripts.evaluate_gmm
bash runs/plotting/components.sh
bash runs/plotting/good_runs.sh
```

All three default to the main experiment. Optional comparisons:

```bash
bash runs/all.sh                    # 00–04, using 1 pb^-1 groups
bash runs/gmm.sh ls_no_group        # 03 with individual LS
bash runs/plotting/mes.sh            # Original normalized ME plots
bash runs/plotting/luminosity.sh     # Recorded luminosity / Global Muon counts
```

`runs/train.sh MODEL ...` and `runs/predict.sh CHECKPOINT DATA ...` remain
available for individual jobs. Metadata download recipes are separate under
`runs/data/`. On macOS, `caffeinate -i bash runs/gmm.sh` prevents idle sleep
with the lid open.

## Data and model

Train/Validation use RR **Muon Good**, DCS-pass and Muon-certified JSON LS.
Nominal and low-PU JSONs are combined. Physical runs have reproducible,
approximately 70/15/15 Train/Validation/Test roles, stratified by era.
Quality-failing LS go to Test. Monitoring and threshold calibration use
separate Validation runs. Zero Global Muon eta entries are excluded.

The main mode sums consecutive LS until **1 pb^-1 of OMS recorded luminosity**
is reached, before extracting features. Groups never cross dataset, run,
missing-LS, split or quality boundaries. A short tail joins the previous
completed group in its segment; a whole segment below the target is excluded.
Comparisons retain `ls_no_group`, `ls_group_1000glb` and `ls_group_7pb`.

The main AE represents each ME with 16 Gaussian weights, means and log-widths.
It hides one complete ME and predicts that histogram from the others:
**392 → 64 → 8 → 64 → 384** (384 Gaussian parameters + 8 visibility bits).
The Train Gaussian reference initializes LS Prefits, whose weights, means
and widths are free. Minimum sigma is 1/√12 native bin; phi wraps periodically.
Sixty fitting steps are a budget, not a guarantee of convergence.

For native bin counts `n_b`, the prediction is `mu_b = N*q_b`, where
`N = sum(n_b)` and `q_b` is the normalized predicted histogram. Both training
and scoring use Input–Predict count deviance:

```text
D_ME = 2 * sum(n_b * log(n_b / mu_b) - n_b + mu_b)
loss = batch mean of the hidden-ME D_ME
score = max(D_ME over observed MEs)
```

Empty MEs are masked. Conditioning on N tests shape, not absolute yield.
Luminosity sets group boundaries; no OMS values or detector-status flags
are model inputs. The threshold is the **99th percentile of held-out Good
calibration groups**. Plot scores are divided by this threshold; the cut is 1.
A group-level calibration rate is not an original-LS-weighted false-alarm rate.

## Code and outputs

```text
scripts/
├── data/           # Cached DIALS loading, metadata and LS grouping
├── models/         # Experiment settings, preprocessing, Gaussian fits and NN
├── train.py        # Fit the selected model and calibrate its threshold
├── predict.py      # Frozen inference on the cached histograms
├── evaluate_gmm.py # Score validation, summaries and reference comparison
└── plotting/       # Scores, components, statistical checks and Good-run study
runs/               # Short shell recipes
```

`data/dials/` holds the source Parquet histograms; `data/{oms,rr,dcs}/` holds
metadata. Models and tables mirror `output/{model,predict}/{LS_MODE}/{MODEL}/`.
Derived Gaussian fits live in `output/dataset/prefit/`; source histograms stay
in `data/dials/`.
The main plot directory is:

```text
output/plots/ls_group_1pb/03_ae_gmm_deviance/
├── test.png, test_log.png, distributions*.png, confusion_matrix.png, ...
├── stat_score_*.png, loss.png
├── {flagged,non_flagged}_{good,json_bad,dcs_bad,rr_bad}/
├── checks/
│   ├── pvalue_cdf.png
│   ├── count_dependence.png
│   └── chi2_vs_toys.png
└── good_run_study/
    ├── overview.png, rpc_status.png, gem_status.png
    └── examples/    # Named run/ME examples and startup comparisons
```

Gaussian case plots show Input/Prefit, Input/Predict and the signed square root
of each bin's deviance. Its squared sum is the ME deviance; it is not a
standard-normal pull. Scores use all native bins even when the display is cropped.
`cases.csv` records which case categories are available and their selected LS.

## Statistical checks

These plots check the **raw per-ME count model**, separately from alarm performance.

| Plot | Question and interpretation |
|---|---|
| `pvalue_cdf.png` | Are nominal per-ME p-values uniform on Test Good? The diagonal is an ideal reference, not a guaranteed expectation for the learned model. |
| `count_dependence.png` | How do deviance/df, the fraction with nominal p < 0.01, and sparse-bin fraction change with each ME's native count? Uses count quantiles, not one broad high-count bin. |
| `chi2_vs_toys.png` | Does the chi-square approximation agree with fixed-Predict multinomial toys on a small Test subset? Shows the finite toy resolution explicitly. |

The nominal chi-square approximation uses `native_bins - 1` degrees of freedom.
Large total counts do not ensure that each bin is well populated. NN prediction
error, shared events and sparse bins can invalidate this approximation.
The max across MEs is not itself one chi-square statistic or a combined p-value.

The toy check holds the predicted histogram fixed. It does not include NN
uncertainty or prove that a run is Good/Bad. With 255 toys, zero exceedances
produce an estimated p of 1/256; this is not a precise small p-value or an upper
confidence bound. Alarm thresholds continue to come from actual Good calibration.
The corresponding tables are `checks/count_summary.csv` and `checks/toys.csv`
in the prediction directory.

## Good-run investigation

`good_run_study/` explains selected high scores among **labelled Good** Test data.
It is a fixed historical case study, not a new classifier or an automatic cause finder.

| Case | Main observation |
|---|---|
| 395729 | Hit-distribution change; RR RPC BAD for the full run despite Muon GOOD. |
| 393168 | Transient hit change; RPC GOOD/STANDBY mixed at run level. |
| 391699 | Very low luminosity, 886 LS in one group, unusual Standalone composition. |
| 395098 / 396222 | First accepted groups; Global-hit context changes drive unstable Standalone predictions. |
| 384382 | A persistent pT-shape change; HLT/L1T issues are possible context, not a proven cause. |
| 2026C | Large accumulated counts amplify small low-hit-bin prediction mismatches. |

`Selected cases` in the overview means 391699, the two startup groups and
384382 LS 194–274; it is a plotting selection, not a shared physical category.
RPC/GEM colors use **whole-run RR summaries**. Mixed does not identify the
status of a particular LS. Muon Good does not imply every Muon subsystem is
Good. The metadata associations are evidence for investigation, not proof of causality.

Reproducible tables are under the main prediction directory's `good_run_study/`:
`examples.csv`, `run_status.csv` and `startup_sensitivity.csv`. The last replaces
context only while holding the observed Standalone target fixed.

## Retained comparisons

| Experiment | Loss / score |
|---|---|
| `00_ref_score` | Input versus fixed Train reference; Validation-normalized max |
| `01_ae_2feat` | Signed mean/std AE; mean standardized reconstruction error |
| `02_ae_gmm_mse` | Input–Predict native-bin MSE with fixed Train bin weights; max over MEs |
| `03_ae_gmm_deviance` | Main model: raw count deviance; max over MEs |
| `04_ae_gmm_deviance_norm_me_bin` | Deviance with fixed bin/ME weights from frozen 03 Good-Train errors; max over MEs |

03 and 04 share raw count-statistic diagnostics; 04's weighted score is never
passed to a chi-square p-value formula. All experiments keep the same splits,
seed 42 and 99th-percentile calibration rule. Existing comparison results remain
available; model selection here is exploratory and based on one seed.
