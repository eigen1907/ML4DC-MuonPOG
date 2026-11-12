# ML4DC-MuonPOG

Small Muon DQM anomaly-detection study. Metadata collection, cached histogram inspection and model experiments are separate steps.

## Workflow

```text
runs/
  01_metadata/   DCS, OMS and Run Registry collection
  02_data/       cached ME plots
  03_models/     five existing AE comparisons
```

```bash
bash runs/01_metadata/dcs.sh
bash runs/01_metadata/rr.sh --years 2025        # lxplus: CERN Kerberos/OTP
bash runs/01_metadata/oms.sh --runs 393240 396153 397332
bash runs/02_data/plot_mes.sh
bash runs/03_models/study.sh --plots-only      # redraw preserved results
uv run --locked python -m unittest discover -s tests
```

The metadata recipes use `.venv-metadata` with only acquisition dependencies. OMS uses personal CERN browser/device login. RR needs a valid Kerberos ticket (`klist -s`) and may prompt for OTP. Authentication caches stay outside the repository. Historical DIALS download scripts have been removed; existing histogram caches are preserved.

## Metadata

One entry point: `scripts/fetch_metadata.py dcs|oms|rr`.

```text
data/{dcs,oms,rr}/
  raw/<year>.json       annual API data
  summary/<year>.csv    annual table for inspection and analysis
```

| Source | Summary contents |
|---|---|
| DCS | Run and inclusive first/last LS of tracker-ready intervals |
| OMS | One row per LS: run/fill, time, LS luminosity, pile-up, stable beams, CMS active, colliding bunches and Muon readiness bits |
| RR | One row per run and one column per main component: good/bad/unknown |

OMS reads LS data and fill bunch counts; the unused run endpoint is no longer queried. Fill responses are stored once per year and joined by each LS's fill number. Only `*_per_lumisection` luminosity fields enter the CSV; plain luminosity fields can be cumulative. New OMS requests select the required attributes through the documented API `fields` parameter.

OMS and RR raw JSONs combine API responses into one annual file, retaining their returned field values; they are merged JSON, not byte-for-byte HTTP archives. Existing migrated OMS data can contain extra fields from earlier queries. There are no source/checksum sidecars or separate reference snapshots. Public DCS JSONs describe tracker readiness; Muon readiness bits come separately from OMS.

RR defaults to the yearly PromptReco query (`--years 2025`). Its CSV has run_number plus cms, muon, dt, csc, rpc, gem, tracker_pixel, tracker_strip, tracking, ecal, hcal, hlt, l1t and lumi columns. Any BAD LS makes that component bad; exclusively GOOD makes it good; other or missing states remain unknown. This table supplies run labels, not the locations of bad LS. Model labels use the muon column from the annual CSVs.

OMS accepts `--runs`; otherwise it uses the RR run list restricted by the cached Muon0 catalogue when available. Missing runs are skipped. Downloads are combined in memory and written once per year. Add `--offline --years 2025` to rebuild an annual CSV without login. The current OMS 2025 file contains three sampled runs, not the full year. Among the 744 cached 2025 runs, RR provides 560 Good and 17 Bad; 167 are unassessed or missing and do not train or calibrate the models.

The study reads the ordinary `data/dcs/raw/2025.json`. Its bytes match the former reference, so removing the duplicate does not change the DCS selection.

## Existing model experiments

| Directory | Comparison |
|---|---|
| `01_ae_2feat` | Preserved original mean/std AE |
| `02_ae_2feat_dcs` | Mean/std AE with DCS-selected Good training LS |
| `03_ae_hist_dcs` | Native-histogram MSE AE |
| `04_ae_dm_k1_dcs` | K=1 Dirichlet-multinomial AE with variance-weighted residuals |
| `05_ae_dm_k1_weighted_window` | Stage 04 score averaged over a centered five-LS window |

Eight MEs are used; OneHLT is excluded. There is no minimum-entry cut. Current training uses RR Good runs split by run into 70/15/15%, seed 42; Bad and unknown are test-only. Validation has separate monitoring/calibration runs, and finite Good calibration scores set the fixed 97.5th-percentile threshold. The five-LS window uses future LS and is an offline diagnostic.

Existing checkpoints, scores, cuts and splits were produced with the earlier provisional labels and remain preserved. Redrawn plots use current RR labels, including unknown; historical labels are only a fallback when the RR summary is absent. Fresh RR labels change training cohorts: the runner rejects incompatible checkpoints before writing results. A new fit needs an explicitly separate output/checkpoint choice; this cleanup does not retrain the models. Run labels are not LS-level truth, and reused Test results remain exploratory.

## Files

- `scripts/fetch_metadata.py`: metadata download and CSV extraction.
- `scripts/data.py`: cached histogram loading, moments, RR labels and splits.
- `scripts/models.py`: feature and histogram AEs.
- `scripts/scoring.py`: residuals, calibration and LS windows.
- `scripts/study.py`: explicit five-experiment runner.
- `scripts/plot.py`, `plots.py`, `diagnostics.py`: raw MEs and model diagnostics.
- `scripts/config.py`: shared study choices.

`runs/` contains commands that perform real work. `tests/` contains small synthetic checks for pagination, raw preservation, labels, run separation and scoring; it does not download data or submit jobs.

Generated metadata, caches, checkpoints and plots are ignored by Git. Existing histogram data is under `output/dataset/raw/`; results use `output/{model,predict,plots}/`. Dataset/provenance catalogues are under `output/dataset/catalogue/`.

## DQMIO production status

Muon0 AOD is available for 2023–2026. The inspected global DBS catalogue has no full RECO tier; RAW-RECO datasets are skims. An AOD-compatible Muon sequence can produce DQMIO, but complete reproduction of every `Muons/` 1D/2D ME is still unverified because some inputs may be missing.

The next production step is one AOD file's product/reference check and a short discovery job. Enumerate actual booked `Muons/` paths, configure per-LS saving, and verify ME coverage and noncumulative contents across two LS before scaling up. No DQMIO production job has run yet.
