# ML4DC-MuonPOG

Small LS-level Muon DQM anomaly-detection study using cached DIALS histograms and CERN metadata.

## Run

```bash
bash runs/01_metadata/dcs.sh
bash runs/01_metadata/rr.sh --years 2025        # CERN Kerberos/OTP
bash runs/01_metadata/oms.sh --runs 393240 396153 397332
bash runs/02_data/plot_mes.sh
bash runs/03_models/study.sh --plots-only      # redraw saved results
bash runs/03_models/study.sh                   # fit or reuse compatible checkpoints
```

Metadata recipes use `.venv-metadata` with only acquisition dependencies. OMS uses CERN device login; RR needs a Kerberos ticket and may prompt for OTP. Authentication caches stay outside the repository. Model recipes use `uv` and the existing local histogram cache; they do not download DIALS data.

## Files

```text
data/
  dcs/<year>.json                  tracker-ready run/LS intervals
  oms/<year>.json                  selected LS conditions
  rr/<year>.json                   component quality counts per run
output/
  dataset/dials/<dataset>/<run>/   ME Parquet files
  model/<experiment>/             checkpoints
  predict/<experiment>/           scores.parquet and metrics.json
  plots/{mes,<experiment>}/       ME and model plots
scripts/                          Python code
runs/{01_metadata,02_data,03_models}/
```

Dataset directory names encode `/` as `__`. Run lists and LS counts are read from the cache itself; no inventory or run-list file is needed. Generated data and results are ignored by Git.

- `fetch_metadata.py`: DCS, OMS and RR downloads to annual JSON files.
- `config.py`, `data.py`: study choices, cache discovery, ME loading, labels and splits.
- `models.py`, `scoring.py`, `study.py`: model fitting, calibration and experiment execution.
- `plot.py`, `plots.py`: ordinary ME plots and individual model score/loss plots.

## Metadata

DCS preserves the public JSON bytes and describes tracker readiness, not Muon certification. OMS stores nine LS attributes: run/LS, start/end time, delivered/recorded LS luminosity, pile-up, stable beams and CMS active. Run/Fill queries are omitted. The current OMS 2025 file contains three sampled runs; OMS is not a model input yet.

RR stores run number, deleted status and numeric quality counts for 14 components. Model labels are derived directly from Muon counts: any BAD gives Bad, exclusively GOOD gives Good, and deleted, incomplete or missing assessments remain Unknown. Counts do not retain the locations of bad LS. OMS uses `--runs` or the saved RR run list, restricted to cached Muon0 runs when available.

## Experiments

| Directory | Model |
|---|---|
| `01_ae_2feat` | Preserved historical mean/std AE |
| `02_ae_2feat_dcs` | Mean/std AE with DCS-selected Good training LS |
| `03_ae_hist_dcs` | Full-histogram MSE AE |
| `04_ae_dm_k1_dcs` | K=1 Dirichlet-multinomial AE with variance-weighted residuals |
| `05_ae_dm_k1_weighted_window` | Stage 04 scores averaged over a centered five-LS window |

Eight MEs are used; OneHLT is excluded. There is no entry cut. RR Good runs are split 70/15/15% by run with seed 42; Bad and Unknown are test-only. Validation is separated into monitoring and calibration runs. The fixed threshold is the 97.5th percentile of finite Good calibration scores. The centered window uses future LS and is an offline study.

Saved checkpoints, scores and splits remain historical results from provisional labels. Redrawing applies current RR display labels; fitting rejects incompatible checkpoints rather than overwriting them. Run labels are not LS-level truth, and the reused Test set remains exploratory.
