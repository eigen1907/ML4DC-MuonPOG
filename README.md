# ML4DC-MuonPOG

A small per-lumisection anomaly study using the 10 one-dimensional `Muons/` monitoring elements in the DIALS `muo` workspace. The experiment uses 2025 `Muon0` PromptReco runs.

## Run

```bash
uv sync
bash runs/01_example.sh      # fetch one ME/run and plot its raw histogram
bash runs/02_all_2025.sh     # collect 2025 Muon0 data; resumable, but lengthy
bash runs/03_ae_base.sh      # train ae_base and plot LS scores
bash runs/04_ae_bin.sh       # train ae_bin and plot scores plus ME reconstructions
bash runs/05_me_plots.sh     # plot raw MEs and ae_base feature reconstructions
```

The first DIALS query requests CERN authentication when needed. Run collection before the experiment on a fresh clone. The collector keeps all 2025 Muon0 processing versions except non-PromptReco 2025G tests; the experiment selects PromptReco only.

## Data and experiment

`data/run_labels.csv` is the **tracked input**: one row per PromptReco run (`run_number,label`). Its 744 runs comprise 727 provisional `good`, 8 `bad_well_known`, and 9 `bad_ambiguous`. Labels describe whole runs, not individual LS.

The collector writes the DIALS run list to `output/dataset/runs.csv` and one Parquet per dataset/run/ME to `output/dataset/raw/`. The experiment selects PromptReco runs and checks them against the tracked labels. Good runs are split by run within each era: about 70% train, 15% validation, and 15% test. Both Bad groups join the test set. Only Good training LS fit the models.

`ae_base` uses two summaries from each ME (`log1p(entries)` and mean histogram position), giving 20 inputs to a 20→8→3→8→20 autoencoder. For the two sparse OneHLT MEs, it pulls the shape mean toward the Good-train pooled mean using 10 in-range-bin pseudo-counts; `log1p(entries)` remains unchanged. The scaler fits raw Good-train features before this adjustment so that standardization does not undo it. Its threshold is the 95th percentile of Good validation LS scores.

`ae_bin` takes all 1,080 stored bins from the same 10 MEs. It uses `ae_base` scores for the shared run split, fits on 20,000 sampled Good training LS, and checks 5,000 Good validation LS. Its score balances the MEs and adjusts for five `eta_entries` ranges using all Good validation LS; the final threshold is **1 for every LS**. It also plots original, predicted, and residual bin contents for one representative run from each test group.

Both AE experiments save LS score views as `train.png`, `val.png`, `test.png`, and `2025.png` (all splits), plus one plot per era.

`05_me_plots.sh` plots all 10 MEs in seven test runs after `ae_base` training. The representative runs are Good 397332 and Bad 396153/395723; Good test runs 392251, 393240, 396763, and 398902 sample the four marked high-score regions. They were selected using offline eta statistics within those regions, not the much sparser OneHLT statistics. Raw LS-by-bin heatmaps are saved as `output/plots/mes/<group>/<run>/<ME>.png`. The model feature plots are saved as `output/plots/ae_base/{input,output,diff}/{stat,shape}/<group>/<run>/<ME>.png`, with a gallery at `output/plots/ae_base/plots.md`. Here `output` is the AE reconstruction and `diff` is input minus output. Shape input is normalized mean bin position (including the OneHLT adjustment) on a 0–1 scale; reconstructions may leave that range. Stat is Good-train-standardized `log1p(entries)`.

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
| `output/plots/mes/` | Raw per-LS ME heatmaps. |
| `output/plots/ae_base/` | LS score plots, feature inputs/reconstructions/differences, and gallery. |
| `output/plots/ae_bin/` | LS score and bin reconstruction plots. |

Everything under `output/` is generated and ignored by Git; `.gitkeep` preserves its four top-level directories.
