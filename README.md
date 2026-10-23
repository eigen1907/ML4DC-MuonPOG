# ML4DC-MuonPOG

A small per-lumisection anomaly study using the 10 one-dimensional `Muons/` monitoring elements in the DIALS `muo` workspace. The experiment uses 2025 `Muon0` PromptReco runs.

## Run

```bash
uv sync
bash runs/01_example.sh      # fetch one ME/run and plot its raw histogram
bash runs/02_all_2025.sh     # collect 2025 Muon0 data; resumable, but lengthy
bash runs/03_ae_base.sh      # train ae_base and plot LS scores
bash runs/04_me_plots.sh     # plot raw MEs and ae_base feature reconstructions
```

The first DIALS query requests CERN authentication when needed. Run collection before the experiment on a fresh clone. The collector keeps all 2025 Muon0 processing versions except non-PromptReco 2025G tests; the experiment selects PromptReco only.

## Data and experiment

`data/run_labels.csv` is the **tracked input**: one row per PromptReco run (`run_number,label`). Its 744 runs comprise 727 provisional `good`, 8 `bad_well_known`, and 9 `bad_ambiguous`. Labels describe whole runs, not individual LS.

The collector writes the DIALS run list to `output/dataset/runs.csv` and one Parquet per dataset/run/ME to `output/dataset/raw/`. The experiment selects PromptReco runs and checks them against the tracked labels. Good runs are split by run within each era: about 70% train, 15% validation, and 15% test. Both Bad groups join the test set. Only Good training LS fit the models.

`ae_base` uses histogram mean and population standard deviation from eight MEs, excluding the two `MuonRecoOneHLT/` MEs. All ten MEs remain cached and available in raw ME and entries plots. Bin centers are normalized as `(center - xmin) / (xmax - xmin)`; η retains the original `abs(center) / max(abs(xmin), abs(xmax))` definition. Mean and std are the count-weighted moments of those positions; there is no low-stat correction or reference histogram. Empty histograms use mean = std = 0 when no distribution is observed. Entries remain diagnostic metadata, not AE inputs.

The 16 features are standardized using Good training LS and feed a 16→8→4→8→16 autoencoder (seed 42, 80 epochs). The LS anomaly score is the mean squared reconstruction error of mean/std in standardized feature units. One fixed threshold is set at the 95th percentile of Good validation LS scores.

The experiment saves LS score views as `train.png`, `val.png`, `test.png`, and `2025.png` (all splits), along with `loss.png` and `distributions.png`. It does not make period-specific score plots.

`04_me_plots.sh` makes seven plot cases from 19 test runs: standalone Good 397332 and Bad 396153/395723, plus four combined provisional-Good regions A–D from the dense elevated clumps selected on the previous mean-only test plot. A (2025B) combines 10 runs/610 LS, B (2025C) 1/4,777, C (2025F) 4/1,507, and D (2025G) 1/428. Their member runs are listed in the gallery. The 16 selected runs contain 7,322 LS. AE mean/std/stat region plots retain **all** LS in run order, including below-threshold and low/zero-stat LS, with run-number ticks.

For controls, raw ME heatmaps are saved as `output/plots/mes/{good,bad}/<run>/<ME>.png`; for regions, as `output/plots/mes/good_but_anomaly/<A|B|C|D>/<ME>.png`. They show stored bin counts for recorded LS. Single-run heatmaps keep actual LS numbers and missing-LS gaps; region heatmaps concatenate all recorded LS with run-number ticks. The corresponding plots use the same suffix under `output/plots/ae_base/{stat,mean,std}/`, with a gallery at `output/plots/ae_base/plots.md`. All ten MEs have raw ME and stat plots; only the eight AE inputs have mean/std reconstruction plots. Mean/std figures overlay the input and AE reconstruction as LS points in a large upper panel, with input minus reconstruction in a smaller lower panel. Mean and std use normalized-position units (0–1 and 0–0.5 respectively); AE outputs may leave those ranges. Stat figures show raw entries in a single panel for diagnosis, without AE predictions or residuals.

The diff panels use linear axes with fixed zoom ranges across runs and MEs: mean `[-0.1, 0.1]` and std `[-0.02, 0.02]`. Residuals outside these ranges are indicated by up/down triangles near the panel boundaries; the upper panel still shows the full input and output values.

Run labels are provisional and apply to entire runs. Flagged-LS fractions are therefore exploratory, not LS classification metrics.

## Files

| Path | Purpose |
|---|---|
| `data/run_labels.csv` | Tracked run labels; the only input needed in Git. |
| `scripts/` | DIALS retrieval, raw ME plots, and the ae_base experiment. |
| `runs/` | Four short bash recipes. |
| `output/dataset/` | Generated run list and cached per-LS ME Parquet files. |
| `output/model/ae_base/` | Fitted `model.pkl`. |
| `output/predict/ae_base/` | LS `scores.parquet` and `metrics.json`. |
| `output/plots/mes/` | Raw per-LS ME heatmaps. |
| `output/plots/ae_base/` | LS scores, mean/std reconstructions, diagnostic entries, and gallery. |

Everything under `output/` is generated and ignored by Git; `.gitkeep` preserves its four top-level directories.
