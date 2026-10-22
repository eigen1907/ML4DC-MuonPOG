# ML4DC-MuonPOG

A small per-lumisection anomaly study using the 10 one-dimensional `Muons/` monitoring elements in the DIALS `muo` workspace. The experiment uses 2025 `Muon0` PromptReco runs.

## Run

```bash
uv sync
bash runs/01_example.sh      # fetch one ME/run and plot its LS map
bash runs/02_all_2025.sh     # collect 2025 Muon0 data; resumable, but lengthy
bash runs/03_ae_base.sh      # train ae_base and plot LS scores
bash runs/04_ae_bin.sh       # train ae_bin and plot scores plus ME reconstructions
bash runs/05_me_plots.sh     # plot LS maps for 10 MEs in three representative runs
```

The first DIALS query requests CERN authentication when needed. Run collection before the experiment on a fresh clone. The collector keeps all 2025 Muon0 processing versions except non-PromptReco 2025G tests; the experiment selects PromptReco only.

## Data and experiment

`data/run_labels.csv` is the **tracked input**: one row per PromptReco run (`run_number,label`). Its 744 runs comprise 727 provisional `good`, 8 `bad_well_known`, and 9 `bad_ambiguous`. Labels describe whole runs, not individual LS.

The collector writes the DIALS run list to `output/dataset/runs.csv` and one Parquet per dataset/run/ME to `output/dataset/raw/`. The experiment selects PromptReco runs and checks them against the tracked labels. Good runs are split by run within each era: about 70% train, 15% validation, and 15% test. Both Bad groups join the test set. Only Good training LS fit the models.

`ae_base` uses two summaries from each ME (`log1p(entries)` and mean histogram position), giving 20 inputs to a 20→8→3→8→20 autoencoder. Its threshold is the 95th percentile of Good validation LS scores.

`ae_bin` takes all 1,080 stored bins from the same 10 MEs. It uses `ae_base` scores for the shared run split, fits on 20,000 sampled Good training LS, and checks 5,000 Good validation LS. Its score balances the MEs and adjusts for five `eta_entries` ranges using all Good validation LS; the final threshold is **1 for every LS**. It also plots original, predicted, and residual bin contents for one representative run from each test group.

Both AE experiments save LS score views as `train.png`, `val.png`, `test.png`, and `2025.png` (all splits), plus one plot per era.

`05_me_plots.sh` draws LS-by-bin heatmaps for all 10 MEs in three test runs: Good 397332 and Bad 396153/395723 (one from each tracked Bad subtype). Each has the highest median per-LS `GlbMuon_Glb_eta` entries in its original group. Each run of the recipe refreshes `output/plots/mes/`, saving one `<ME>.png` directly under each dataset/run directory; its `plots.md` links the figures independently of the AE prediction plots. Plots display the Bad subtypes together and put `Run number (Good/Bad), era` above each single-run ME view.

Run labels are provisional and apply to entire runs. Flagged-LS fractions are therefore exploratory, not LS classification metrics; the bin score's occupancy adjustment can also suppress count-only anomalies.

## Files

| Path | Purpose |
|---|---|
| `data/run_labels.csv` | Tracked run labels; the only input needed in Git. |
| `scripts/` | DIALS retrieval, raw ME plots, and the two AE experiments. |
| `runs/` | Five short bash recipes. |
| `output/dataset/` | Generated run list and cached per-LS ME Parquet files. |
| `output/model/{ae_base,ae_bin}/` | Fitted `model.pkl` files. |
| `output/predict/{ae_base,ae_bin}/` | LS `scores.parquet` and `metrics.json`. |
| `output/plots/mes/` | Model-independent LS-by-bin ME heatmaps. |
| `output/plots/{ae_base,ae_bin}/` | Score plots; `ae_bin` also has reconstruction figures and `plots.md`. |

Everything under `output/` is generated and ignored by Git; `.gitkeep` preserves its four top-level directories.
