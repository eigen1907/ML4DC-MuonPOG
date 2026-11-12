# ML4DC-MuonPOG

Small LS-level Muon anomaly-detection study using cached DIALS histograms and CERN metadata.

## Workflow

```text
Cached MEs + metadata → preprocessing → train → checkpoint → predict → plot
```

Training saves the model, fitted scaling and a fixed Good-calibration threshold. Prediction uses that checkpoint without fitting or recalibrating. Plotting reads saved predictions independently.

Test plots color LS points by the run's Good/Bad/Unknown assessment. Points follow consecutive LS, with a few horizontal Run ticks. Score means anomaly score divided by the fixed threshold; the threshold line is therefore 1. Thin lines mark every run boundary.

```bash
# Check DIALS availability without downloading histograms
bash runs/data/dials.sh --years 2024 2025 2026 --list-only

# Fetch one run first, or all available Muon0 processing versions
bash runs/data/dials.sh --years 2025 --run 397332
bash runs/data/dials.sh --years 2024 2025 2026

# Annual metadata; OMS remains a bounded sample
bash runs/data/dcs.sh --years 2024 2025 2026
bash runs/data/rr.sh --years 2024 2025 2026       # CERN Kerberos/OTP
bash runs/data/oms.sh --runs 393240 396153 397332  # CERN device login

# Inspect cached histograms
bash runs/plotting/mes.sh
```

For the 2024–2026 study, train on 2024–2025. The combined Test contains held-out
Good runs plus all Bad/Unknown runs from 2024–2025, followed by all 2026 runs.
Prediction files remain separate; the plotter combines them using the same
checkpoint and fixed threshold:

```bash
bash runs/train.sh model_03_ae_dm_k1 --years 2024 2025 --output output/model/2024_2025/03_ae_dm_k1
bash runs/predict.sh output/model/2024_2025/03_ae_dm_k1 data/dials output/predict/2024_2025/03_ae_dm_k1 --years 2024 2025 --prompt-reco
bash runs/predict.sh output/model/2024_2025/03_ae_dm_k1 data/dials output/predict/2026/03_ae_dm_k1 --years 2026 --prompt-reco
# Redraw the combined Test independently of training and prediction
bash runs/plotting/scores.sh --input output/predict/2024_2025/03_ae_dm_k1 output/predict/2026/03_ae_dm_k1 --output output/plots/03_ae_dm_k1
```

The Good split remains 70/15/15% by run, stratified by era with seed 42. No 2026
data enter training, scaling, epoch selection or calibration. Use `--prompt-reco`
for prediction to match the training cohort and avoid adding other processings.
The cached 2024 C–E reprocessings are retained but excluded from this study.
With the current RR snapshot, 2026 has no Muon Bad labels: it tests stability on
Good runs and allows inspection of Unknown runs, not Bad detection efficiency.
Multiple plot inputs must share the checkpoint, threshold and training signature;
only Test rows are included, and an explicit `--output` directory is required.
`output/plots/` contains only `mes/` and the numbered model folders. Each
model folder holds the combined 2024–2026 Test views, distributions and confusion
matrix, plus the 2024–2025 training loss curve. No year subfolders are used.
Each model also gets `confusion_matrix.png`: DCS-selected, finite-score Good/Bad
Test LS, with counts and row percentages at the fixed threshold. LS inherit their
RR run labels; Unknown/unscored LS are excluded. This is not LS-level ground truth.
`confusion_matrix_no2026C.png` also shows the Test after excluding 2026C, using
the same threshold. Category labels remain; axis titles are omitted.

DIALS collection uses the `muo` workspace and caches available per-LS `Muons/` 1D MEs from every matching Muon0 processing version. Availability is determined by the live listing; requesting a year does not imply that it is present. Existing valid Parquet files are reused.

`runs/train.sh MODEL [OPTIONS]` and `runs/predict.sh CHECKPOINT DATA [OUTPUT] [OPTIONS]` forward options such as `--years 2024 2025 2026`. Both use all cached years when `--years` is omitted. The score plotter accepts optional `--checkpoint` and `--output`; it finds the recorded checkpoint automatically when available.

Window experiments reuse their source weights and calibrate the averaged score;
they do not train another network. Stage 04 uses Stage 03; Stage 05 uses Stage 01.
Their `loss.png` shows the source training history:

```bash
bash runs/train.sh model_04_ae_dm_k1_weighted_window \
  --years 2024 2025 --source output/model/2024_2025/03_ae_dm_k1 \
  --output output/model/2024_2025/04_ae_dm_k1_weighted_window
bash runs/train.sh model_05_ae_2feat_window \
  --years 2024 2025 --source output/model/2024_2025/01_ae_2feat \
  --output output/model/2024_2025/05_ae_2feat_window
```

Metadata recipes use `.venv-metadata`; DIALS collection uses `.venv-dials` and the `dials` dependency group. Model and plotting recipes use the project environment. Credentials stay outside the repository. Commands run from the repository root through `python -m scripts.…`.

## Structure

```text
scripts/
  data/
    metadata.py             download DCS, OMS and RR JSON
    fetch_dials.py          list and cache available Muon0 histograms
    dials.py                read cached MEs, metadata labels and run splits
  models/
    config.py               paths, MEs and shared study choices
    preprocessing.py        moments, histogram inputs and training selections
    core.py                 shared network, residuals, calibration and checkpoints
    model_0*.py             numbered experiment definitions (01–05)
  train.py                  fit/reuse one experiment and save its threshold
  predict.py                apply a saved checkpoint to cached MEs
  plotting/
    mes.py                  normalized ME and entries plots
    scores.py               saved anomaly-score and loss plots
runs/
  data/                     DIALS and metadata download recipes
  train.sh                  MODEL → training/calibrated checkpoint
  predict.sh                checkpoint → predictions
  plotting/                 ME and score plot recipes
data/
  dials/<dataset>/<run>/         cached ME Parquet files
  {dcs,oms,rr}/<year>.json       metadata
output/
  model/2024_2025/<experiment>/      checkpoints and fitted threshold
  predict/{2024_2025,2026}/<experiment>/  scores.parquet and metrics.json
  plots/mes/                        normalized ME figures
  plots/01_ae_2feat/                 combined 2024–2026 Test figures
  plots/02_ae_hist/
  plots/03_ae_dm_k1/
  plots/04_ae_dm_k1_weighted_window/
  plots/05_ae_2feat_window/
```

Dataset folder names encode `/` as `__`. Available runs are discovered from the cache; no inventory is required. Generated data and results are ignored by Git.

## Experiments

| ID / result directory | Change |
|---|---|
| `01_ae_2feat` | Mean/std AE |
| `02_ae_hist` | Full-histogram MSE AE |
| `03_ae_dm_k1` | K=1 Dirichlet-multinomial AE; variance-weighted residuals |
| `04_ae_dm_k1_weighted_window` | Stage 03 scores averaged over a centered five-LS window |
| `05_ae_2feat_window` | Stage 01 mean/std AE scores averaged over the same five-LS window |

All experiments use DCS-selected Good training LS, so their names omit the DCS suffix. `01_ae_2feat` is the former `02_ae_2feat_dcs`; the unfiltered historical baseline has been removed.

Eight MEs are used; OneHLT is excluded. There is no entry cut. Good runs are split 70/15/15% by run with seed 42; Bad and Unknown are test-only. Validation has separate monitoring and calibration runs. The threshold is the 97.5th percentile of finite Good calibration scores. Stages 04–05 average up to two preceding and two following LS without crossing run, dataset, LS-gap or DCS boundaries. Each uses its own threshold from the same Good calibration runs; both are offline studies.

Collecting more years or MEs does not change these model inputs. Training requires one PromptReco processing per physical run and rejects duplicates, even across different dataset folders. Use a cache containing one processing per run for training; prediction can inspect all cached processing versions. Each selected year requires its annual DCS JSON.

Existing checkpoints, predictions and splits are preserved historical results from provisional labels. Current RR labels can change training splits: incompatible checkpoints are rejected, so use a new output directory for a new fit. Prediction also preserves existing output files. Plotting only updates displayed labels; it retains saved scores, splits and thresholds.

## Metadata

- **DCS:** public tracker-ready run/LS intervals, preserving the downloaded JSON bytes. These are not Muon certification labels.
- **OMS:** run/LS, start/end time, delivered/recorded luminosity, pile-up, stable beams and CMS active. Only sampled runs have been collected; extending the DIALS cache does not download OMS automatically. Without `--runs`, OMS selects cached Muon0 runs first, then RR runs if the cache is empty. OMS is not a model input yet.
- **RR:** run number, `rr_dataset`, deleted status and quality counts for 14 components. Nondeleted PromptReco Collisions takes priority, then Commissioning, then online for missing cached Muon0 runs. Online-only records always remain Unknown for training. Otherwise Muon BAD takes precedence; exclusively GOOD gives Good; incomplete, missing or deleted assessments remain Unknown.

RR labels apply to runs and do not locate bad LS. Test results remain exploratory.
