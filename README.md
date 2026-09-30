# ML4DC-MuonPOG

Small per-lumisection study of the 10 one-dimensional `Muons/` monitoring elements in the DIALS `muo` workspace. Set up Python with `uv sync`; the first DIALS query starts CERN device authentication.

```bash
bash runs/01_example.sh      # one run/ME and three raw-data plots
bash runs/02_all_2025.sh     # cache 2025 Muon0; exclude non-PromptReco 2025G tests
bash runs/03_experiment.sh   # fit and score the small PromptReco autoencoder
```

The 2025 collection completed on 30 September 2026. It caches 795 dataset/run pairs (744 distinct runs) and 10 `Muons/` MEs, totaling 7,950 Parquet files. The two non-PromptReco 2025G test datasets are excluded; both C and F PromptReco versions are retained. The recipe skips completed files on restart. For a future background run, open `zellij -s ml4dc-muon0`, run `bash runs/02_all_2025.sh 2>&1 | tee -a data/collect.log` inside, and detach with Ctrl+o then d.

Each raw Parquet holds all available LS for **one exact dataset, run, and ME**. Different processing versions remain separate:

```text
data/raw/Muon0__Run2025D-PromptReco-v1__DQMIO/395719/
  Muons__MuonRecoAnalyzer__GlbMuon_Glb_eta.parquet
```

`data/runs.csv` is the small DIALS-generated dataset/run list. The model uses all 744 PromptReco runs, including C/F v1 and v2; each run appears in exactly one PromptReco dataset. Run IDs from the user-provided `examples/2025.csv` are interpreted as 9 **Bad (ambiguous)**, 8 **Bad (well known)**, and 727 **provisional Good** after resolving overlaps. The file is a three-section note despite its `.csv` name. These are run-level study groups, not certified LS-level truth. `examples/datasets.csv` has Run Registry workflow states such as `OPEN` and `COMPLETED`, which are not Muon quality labels.

The autoencoder uses 20 inputs per LS: `log1p(entries)` and one normalized mean position of the histogram contents for each of the 10 MEs. η uses the absolute bin position so symmetric forward changes remain visible. An empty histogram has shape summary zero; `entries=0` stays zero too. A missing ME file or LS row is never treated as zero. The 727 provisional-Good runs are split **by run within each era** into approximately 70% training, 15% validation, and 15% test. All 8 Bad (well known) and 9 Bad (ambiguous) runs are added to the test set; neither group is used for fitting or threshold choice. A small 20→8→3→8→20 `scikit-learn` autoencoder reconstructs the standardized inputs for 80 epochs. The **anomaly score** is the mean squared difference between its reconstructed and original standardized inputs; larger means less like the training data. The **LS threshold** is one fixed number for all eras: the 99th percentile of provisional-Good validation LS scores. Scores above it are flagged, not automatically certified bad.

`runs/03_experiment.sh` writes `results/model.pkl`, per-LS `results/scores.parquet`, `results/metrics.json`, and plots in `results/plots/`. The Parquet `split` column is `train`, `validation`, or `test`; its `label` column distinguishes `good`, `bad_well_known`, and `bad_ambiguous`. The six era plots (`2025B.png` through `2025G.png`) and `all.png` show every included split; `test.png` shows held-out Good and both Bad groups together. In each score plot, blue LS points are concatenated in run order, while sparse x-axis ticks show run numbers. Green, red, and yellow backgrounds identify provisional Good, Bad (well known), and Bad (ambiguous) **runs**, not LS-specific quality truth; the black dashed line is the shared LS threshold. The score axis is logarithmic so isolated large scores do not hide the bulk of the LS points. `distributions.png` compares the three test-group LS score distributions. `loss.png` shows train and held-out validation reconstruction MSE after each epoch.

`metrics.json` records dataset sizes, reconstruction losses, the fixed LS threshold, and the descriptive fraction of flagged LS in each split/label group. It does not calculate AUROC or TP/FP/FN/TN rates: the available labels apply to runs, not individual LS. The Good labels are provisional. This random within-era run split tests unseen runs from the same year, not future-era performance. DIALS reports incomplete expected LS coverage for 113 of the 744 runs, although every cached ME matches its returned DIALS LS count. Scores can reflect trigger, exposure, or era conditions as well as detector behavior.

| Path | Purpose |
|---|---|
| `scripts/fetch.py`, `scripts/collect.py` | Authenticated single-unit retrieval and resumable full collection. |
| `scripts/plot.py` | Three CMS-style raw-histogram diagnostics for one ME/run. |
| `scripts/experiment.py` | PromptReco feature loading, run split, autoencoder, and LS score plots. |
| `runs/` | Three fixed bash recipes. |
| `examples/` | User-provided run notes and Run Registry export. |
| `data/`, `plots/`, `results/` | Ignored raw cache, raw diagnostics, and experiment outputs. |

`PROJECT_DEFINITION.md` preserves the initial reference study and earlier project plan.
