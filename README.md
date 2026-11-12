# ML4DC-MuonPOG

Small LS-level Muon anomaly-detection study using cached DIALS histograms and CERN metadata.

## Workflow

```text
Cached MEs + metadata → preprocessing → train → checkpoint → predict → plot
```

Training saves the model, fitted scaling and a fixed Good-calibration threshold. Prediction uses that checkpoint without fitting or recalibrating. Plotting reads saved predictions independently.

```bash
# Optional metadata updates; no DIALS download
bash runs/data/dcs.sh --years 2025
bash runs/data/rr.sh --years 2025                 # CERN Kerberos/OTP
bash runs/data/oms.sh --runs 393240 396153 397332  # CERN device login

# Inspect cached histograms
bash runs/plotting/mes.sh

# Train one experiment into a new directory, then predict and plot
bash runs/train.sh model_03_ae_dm_k1 --output output/model/dm_rr
bash runs/predict.sh output/model/dm_rr data/dials output/predict/dm_rr
bash runs/plotting/scores.sh --input output/predict/dm_rr

# Redraw an existing result without training or prediction
bash runs/plotting/scores.sh --input output/predict/03_ae_dm_k1
```

`runs/train.sh MODEL [OPTIONS]` takes a model module basename, such as `model_03_ae_dm_k1`, and forwards `--data`, `--output` and `--source` to training. `runs/predict.sh` takes `CHECKPOINT DATA [OUTPUT]`. The score plotter accepts optional `--checkpoint` and `--output`; it finds the recorded checkpoint automatically when available.

Stage 04 uses Stage 03 weights and calibrates the window score; it does not train another network:

```bash
bash runs/train.sh model_04_ae_dm_k1_weighted_window \
  --source output/model/dm_rr --output output/model/dm_window_rr
```

Metadata recipes use `.venv-metadata`; other recipes use the project environment. Credentials stay outside the repository. Commands run from the repository root through `python -m scripts.…`.

## Structure

```text
scripts/
  data/
    metadata.py             download DCS, OMS and RR JSON
    dials.py                read cached MEs, metadata labels and run splits
  models/
    config.py               paths, MEs and shared study choices
    preprocessing.py        moments, histogram inputs and training selections
    core.py                 shared network, residuals, calibration and checkpoints
    model_0*.py             numbered experiment definitions (01–04)
  train.py                  fit/reuse one experiment and save its threshold
  predict.py                apply a saved checkpoint to cached MEs
  plotting/
    mes.py                  normalized ME and entries plots
    scores.py               saved anomaly-score and loss plots
runs/
  data/                     metadata download recipes
  train.sh                  MODEL → training/calibrated checkpoint
  predict.sh                checkpoint → predictions
  plotting/                 ME and score plot recipes
data/
  dials/<dataset>/<run>/         cached ME Parquet files
  {dcs,oms,rr}/<year>.json       metadata
output/
  model/<experiment>/           checkpoints and fitted threshold
  predict/<experiment>/         scores.parquet and metrics.json
  plots/{mes,<experiment>}/     figures
```

Dataset folder names encode `/` as `__`. Available runs are discovered from the cache; no inventory is required. Generated data and results are ignored by Git.

## Experiments

| ID / result directory | Change |
|---|---|
| `01_ae_2feat` | Mean/std AE |
| `02_ae_hist` | Full-histogram MSE AE |
| `03_ae_dm_k1` | K=1 Dirichlet-multinomial AE; variance-weighted residuals |
| `04_ae_dm_k1_weighted_window` | Stage 03 scores averaged over a centered five-LS window |

All four experiments use DCS-selected Good training LS, so their names omit the DCS suffix. `01_ae_2feat` is the former `02_ae_2feat_dcs`; the unfiltered historical baseline has been removed.

Eight MEs are used; OneHLT is excluded. There is no entry cut. Good runs are split 70/15/15% by run with seed 42; Bad and Unknown are test-only. Validation has separate monitoring and calibration runs. The threshold is the 97.5th percentile of finite Good calibration scores. Stage 04 uses future LS and is an offline study.

Existing checkpoints, predictions and splits are preserved historical results from provisional labels. Current RR labels can change training splits: incompatible checkpoints are rejected, so use a new output directory for a new fit. Prediction also preserves existing output files. Plotting only updates displayed labels; it retains saved scores, splits and thresholds.

## Metadata

- **DCS:** public tracker-ready run/LS intervals, preserving the downloaded JSON bytes. These are not Muon certification labels.
- **OMS:** run/LS, start/end time, delivered/recorded luminosity, pile-up, stable beams and CMS active. The current file contains three sampled runs. OMS is not a model input yet.
- **RR:** run number, deleted status and quality counts for 14 components. Muon BAD takes precedence; exclusively GOOD gives Good; incomplete, missing or deleted assessments remain Unknown.

RR labels apply to runs and do not locate bad LS. Test results remain exploratory.
