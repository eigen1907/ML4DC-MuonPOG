**ML-assisted CMS Muon data certification: initial project definition**  
Reference review: 30 September 2026. This is the original study report; some proposed choices below predate the full Muon0 collection and first autoencoder. See `README.md` for the current workflow and implementation status.

**Start with an offline per-LS anomaly-ranking study for Muon reconstruction.** Raw acquisition can cover all 2025 Muon0 processing datasets and available `Muons/` MEs. Select one coherent dataset/period and three or four MEs for the first model pilot. Produce inspectable scores, diagnostic plots, and comparisons with certification information. The initial result supports expert review; an anomaly score does not establish that a lumisection is unsuitable for physics.

I reviewed all 56 tutorial slides, including the model-deployment and OMS backup material, and inspected the current code in the three reference repositories. Tutorial commands are reference material, not instructions to execute. The principal distinction is between retrieving already-produced per-LS data, developing a local model, and subsequently deploying that model into DIALS. [Tutorial, especially slides 9–21, 26–40, 48–56](</Users/eigen1907/Library/Mobile Documents/com~apple~CloudDocs/Muon_ML4DQMDC/ml4dqm_20260930.pdf>).

**The actual data flow begins in CMSSW.**

```text
CMSSW DQM → selected per-LS MEs in DQMIO → DIALS ingestion/database
    → authenticated cmsdials queries → local raw Parquet cache
    → validated run/LS alignment + optional OMS context
    → features → baseline fit/calibration → held-out per-LS scores
    → plots and independent certification comparison
```

DIALS cannot recover LS detail from a run-integrated histogram. The tutorial describes ingestion of per-LS DQMIO MEs, but actual coverage must be verified for the chosen dataset and processing campaign. If an ME exists per LS in DQMIO but is missing from DIALS, ask about ingestion; if it was only stored per run, historical recovery generally requires reconstruction and is outside this pilot.

The local sequence should be:

1. Authenticate interactively with `Credentials.from_creds_file()` and select the appropriate workspace. The examples explicitly use `tracker`; do not inherit that setting for Muon work without checking.
2. Inspect `dataset_index`, `run`, `lumi`, and `mes`. The ME catalogue supplies names and dimensions, but catalogue membership alone does not prove coverage in a particular run.
3. Request exact dataset, ME, and run selections through `h1d` or `h2d`. Start with `max_pages=1` for a schema check; subsequently fetch all pages for the bounded selection. A truncated response is not a complete dataset.
4. Convert responses with `to_pandas()` and cache them. `dialstools` demonstrates per-run queries and Parquet output split by dataset and ME; adopt this pattern with a small retrieval script. [Retrieval implementation](https://github.com/LukaLambrecht/dialstools/blob/0899b4050a88f91bc0697e043d286f0bbefbec16/datasets/get_data_dials.py).

Retain dataset identity, `run_number`, `ls_number`, ME name/ID, file ID, axis metadata, `entries`, and the histogram payload. Raw records are keyed by dataset/run/LS/ME, with file provenance retained to investigate duplicates. The feature table has one row per dataset/run/LS. Join using keys, never row position; do not silently sum overlapping processing versions or duplicate records.

Store the exact query, workspace, client version, retrieval time, and completion status beside the cache. Compare requested LS coverage with retrieved coverage and retain missing/invalid records in an audit table. A Parquet file existing is not evidence that acquisition completed.

**The first prototype should establish a small, defensible input set.** The current CMSSW per-LS configuration contains these concrete candidates:

| Candidate ME | Initial use |
|---|---|
| `Muons/MuonRecoAnalyzer/GlbMuon_Glb_eta` | Coarse regional fractions and acceptance changes |
| `Muons/MuonRecoAnalyzer/GlbMuon_Glb_phi` | Coarse angular fractions and localized deficits |
| `Muons/globalMuons/GeneralProperties/NumberOfMeanRecHitsPerTrack_glb` | Hit-quality distribution and low-hit fraction |
| `Muons/MuonRecoAnalyzer/GlbMuon_Glb_pt` | Optional fourth input: spectrum/tail changes and sample context |

These paths are verified in the [CMSSW per-LS selection](https://github.com/cms-sw/cmssw/blob/96f3b11c421a3cae7b01f5d985b2f8bef2b90832/DQMServices/Core/python/nanoDQMIO_perLSoutput_cff.py) and are present in the `muo` catalogue. Their statistics and usefulness across the selected runs still need checking. Start with the first three and retain the fourth only if its per-LS statistics justify it. Defer dimuon-mass and isolation MEs until the core inputs work.

This pilot uses only MEs under `Muons/` to study Muon reconstruction quality.

First inspect one run; then assemble roughly 5–10 runs, aiming at a few thousand LS, including several nominal runs and independently documented problematic intervals. Use a single sufficiently populated stream and processing version. Choose Express versus PromptReco after checking coverage, event selection, and whether the intended application is rapid feedback or retrospective certification. Do not assume the tutorial's ZeroBias choice provides enough reconstructed muons.

**OMS should initially explain operating conditions.** Cache stable-beam/CMS-active status and relevant Muon DCS readiness where available; then consider pileup, luminosity, and a trigger rate relevant to the chosen stream. Use these first for eligibility and diagnostic plots. Add numerical OMS features only through a measured DQM-only versus DQM-plus-OMS comparison, so the model does not merely reproduce beam or trigger conditions.

There is a useful update beyond the tutorial: the inspected `cmsdials` 2.2.0 distribution exposes `dials.oms.query(...)`, returning JSON through an OMS proxy. Try this route before adding a separate OMS client; access, endpoints, pagination, and field meanings still need a small authenticated check. Local code must align OMS records by run/LS and preserve missingness. Automatic OMS delivery configured for a deployed DIALS model, shown in slides 53–56, does not perform this local preparation for us. [Published client distribution](https://pypi.org/project/cmsdials/2.2.0/#files).

**Preprocessing must preserve the distinction between shape, rate, and unavailable data.**

- Validate dimensions, axis definitions, finite values, payload lengths, and uniqueness. Confirm flow-bin conventions and 2D orientation before reshaping. Preserve raw payloads; reject incompatible binning rather than silently concatenating it.
- Keep histogram integrals/entries separately from normalized shape fractions. `entries` is not automatically the number of events or an exposure denominator. Normalize rates only when a meaningful denominator and trigger/prescale treatment are established.
- Use coarse, physically meaningful bins and about 10–20 features: angular-region fractions, a hit-quality summary, an optional high-pT fraction, and count summaries. Avoid arbitrary smoothing across LS, which could erase short defects.
- Distinguish absent MEs, low exposure, and genuine zero occupancy. A zero histogram under substantial exposure may itself indicate a problem. Missing required inputs produce an explicit unscored status; operational exclusions and low-statistics cases retain reason codes.
- Fit reference distributions, transformations, and any data-derived masks on training runs only. Persist them with the model. Do not transfer the pixel example's normalization, dead-channel masks, or spatial thresholds to Muon MEs without validation.

**Use one simple learned baseline and one transparent comparator.** Start with per-feature median/MAD deviations on a nominal training sample, reporting the largest deviations as diagnostics. Handle constant features explicitly. Compare this with scikit-learn `IsolationForest` on the same compact feature table, using a fixed seed and modest default settings. Its purpose is to detect unusual combinations with little training infrastructure. Use a documented score convention with larger values meaning more anomalous; calibrate the alert threshold separately. Neither its score nor its `contamination` setting estimates the probability of bad data. [scikit-learn anomaly-detection documentation](https://scikit-learn.org/stable/modules/outlier_detection.html).

If compact summaries demonstrably miss relevant shape changes, the next comparison can be PCA reconstruction residuals, or NMF for suitable nonnegative occupancy maps. The tutorial's NMF example also contains substantial pixel-specific filtering and postprocessing; its success does not establish the right Muon model. Defer neural autoencoders until a simpler baseline exposes a concrete limitation.

**Evaluation must separate certification agreement from anomaly validity.** Use whole runs, preferably whole fills, for chronological training/calibration/test separation. Never randomly split neighboring LS or let the same detector incident enter both calibration and test. Select thresholds on calibration runs against an agreed review/false-alert budget; freeze them before testing.

Use Muon-specific Run Registry information or expert-reviewed incidents when available. Preserve label source, version, granularity, and uncertainty. Official global golden-JSON membership is useful comparison information, but exclusion does not establish a Muon defect, and membership can coexist with subtle anomalies. A run-level flag does not locate an LS-level defect. DIALS model-generated golden-like JSON is another model's output, not independent truth.

Report scored/excluded/missing coverage; flagged fraction on reference-good LS; recall of known bad episodes; and precision/recall curves only where reliable positive and negative labels exist. Include counts and variation across runs, plus luminosity-weighted impact if matched luminosity is available. Treat unmatched labels as unknown. Without trusted bad examples, report ranking and reviewed cases without claiming certification accuracy.

The review bundle should contain score-versus-LS plots with label/DCS overlays, raw and normalized ME comparisons, feature deviations for high-scoring LS, and an annotated shortlist. Inspect some unflagged LS as well. Synthetic distortions may test sensitivity but cannot replace real fault validation.

**Keep the repository small.** Stage 1 contains only the files needed for retrieval and inspection:

```text
ML4DC-MuonPOG/
  README.md
  PROJECT_DEFINITION.md
  pyproject.toml
  uv.lock
  examples/                  # reference notebook and run notes
  scripts/
    fetch.py
    collect.py
    plot.py
  runs/                      # small example and full 2025 download
    01_example.sh
    02_all_2025.sh
  data/                      # ignored: run list and raw Parquet cache
  plots/                     # ignored: diagnostic figures
```

At the initial definition stage, Python 3.12 and `uv` provided the small retrieval environment. The later pilot adds `scikit-learn` without introducing a `src/` package. [uv project files](https://docs.astral.sh/uv/concepts/projects/layout/).

Borrow DeepMuonReco's package/scripts separation, reproducible environment, seed recording, and saved experiment artifacts. Its Hydra hierarchy, deep-learning stack, and batch machinery are unnecessary here. [Inspected project definition](https://github.com/eigen1907/DeepMuonReco/blob/484f02460ab65f4c5c9ec75415616966054f1bd5/pyproject.toml). For plots, adopt `mplhep` CMS styling, readable labels, consistent colors, and small reusable figure/save helpers from the RPC reference; use an appropriate internal-work label and accurate dataset/run annotations. [Inspected plotting utilities](https://github.com/eigen1907/RPCDPGAnalysis/blob/088882aaa906599d1b6e351ff4731a9dbe4b8b1c/NanoAODTnP/python/PlotUtils.py).

**Defer Aim for the first pilot.** Each result directory should already contain settings, input-manifest hash, code revision, seed, model/preprocessing artifact, threshold, metrics JSON, scores Parquet, and plots. Aim becomes worthwhile when repeated model/feature comparisons make these directories cumbersome; it can then log parameters and metrics through a direct `aim.Run` without introducing a logging framework. [Aim run management](https://aimstack.readthedocs.io/en/latest/using/manage_runs.html).

**Resolve these decisions before training.**

| Decision | Proposed default / evidence needed |
|---|---|
| Physics scope | Resolved: only `Muons/` reconstruction MEs |
| Data selection | One era, stream, processing version; confirm workspace and actual ME coverage |
| Usable statistics | Inspect per-LS counts, binning, missingness, and exposure dependence before fixing features |
| Reference sample | Identify nominal runs and independent problematic episodes; document certification granularity |
| Operating objective | Expert triage first; agree on tolerable review load before threshold calibration |
| Access/environment | Confirm device authentication and OMS-proxy access on laptop or lxplus |

At reference-review time, CERN GitLab pages required sign-in and the protected service had not been queried. Stage 1 has since confirmed authenticated `muo` access and one complete run retrieval. The 2.2.0 histogram filter models spell the upper-LS field `ls_numbet__lte`; exact-run retrieval avoids relying on that field.

**Proceed in short stages.**

1. **Access and inventory (complete):** establish the minimal uv environment; discover the workspace/dataset/MEs; fetch one page for one ME/run, then one complete bounded run. The schema/coverage summary and diagnostic plots are described in `README.md`.
2. **Freeze a pilot dataset:** select 3–4 MEs and the small run set; cache DQM, labels, and useful OMS context; validate joins and record training/calibration/test runs. Deliver a reproducible manifest and feature table.
3. **Fit and inspect:** compare the transparent reference score with Isolation Forest, calibrate once, and evaluate on held-out runs. Deliver scores, coverage/label metrics, and annotated cases.
4. **Decide from evidence:** retain, revise, or stop the approach according to statistical sufficiency, review burden, and known-incident sensitivity. Expand the sample, try a shape model, or add Aim only for an identified need. DIALS deployment through `dismcli`, interface/configuration files, registration, and activation is a later project milestone.

Stage 1 demonstrates that Muon per-LS data can be retrieved and inspected reproducibly. The full 2025 raw download is separate from later training-set selection.
