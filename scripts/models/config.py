"""Shared choices for cached inputs, fixed examples and numbered experiments."""

from pathlib import Path


OUT = Path("output")
DIALS = Path("data/dials")
PROMPT_RECO_PATTERN = r"^/Muon0/Run20\d\d[A-Z]-PromptReco-v\d+/DQMIO$"
RR_DATA = Path("data/rr")
DCS_DATA = Path("data/dcs")
IDS = ["dataset", "run_number", "ls_number"]
SEED = 42
LS_MODES = ('ls_no_group', 'ls_group_1000glb', 'ls_group_7pb', 'ls_group_1pb')
SPLIT_POLICY = "rr_dcs_muon_positive_eta_v1"
SPECIAL_RUNS = (398902, 398903)

# Fixed historical examples, not training or threshold selections.
REGIONS = {
    "A": (391550, 391670, 391675, 391677, 391682, 391826, 391906, 391953, 392088, 392090),
    "B": (393240,),
    "C": (396727, 396735, 396757, 396763),
    "D": (398902,),
}
EXAMPLES = {"good": (397332,), "bad": (396153, 395723)}
MODEL_MES = (
    "Muons__Isolation__global__relPFIso_R03",
    "Muons__MuonRecoAnalyzer__GlbMuon_Glb_eta",
    "Muons__MuonRecoAnalyzer__GlbMuon_Glb_phi",
    "Muons__MuonRecoAnalyzer__GlbMuon_Glb_pt",
    "Muons__diMuonHistograms__GlbGlbMuon_HM",
    "Muons__diMuonHistograms__GlbGlbMuon_LM",
    "Muons__globalMuons__GeneralProperties__NumberOfMeanRecHitsPerTrack_glb",
    "Muons__standAloneMuonsUpdatedAtVtx__HitProperties__NumberOfValidRecHitsPerTrack_sta",
)

REFERENCE_SCORE = "00_ref_score"
TWO_FEATURE_AE = "01_ae_2feat"
GMM_MSE = "02_ae_gmm_mse"
GMM_DEVIANCE = "03_ae_gmm_deviance"
GMM_NORM_DEVIANCE = "04_ae_gmm_deviance_norm_me_bin"
MAIN_EXPERIMENT = GMM_DEVIANCE
DEFAULT_LS_MODE = "ls_group_1pb"
MAIN_PREDICTIONS = OUT / "predict" / DEFAULT_LS_MODE / MAIN_EXPERIMENT
EXPERIMENTS = {
    TWO_FEATURE_AE: "Mean/std AE using signed eta; no external model inputs",
    GMM_MSE: "Masked free16 GMM AE with bin-normalized Input–Predict MSE",
    GMM_DEVIANCE: "Masked free16 GMM AE with Input–Predict count deviance loss and score",
    GMM_NORM_DEVIANCE: "Masked free16 GMM AE with frozen Good-Train ME/bin-normalized deviance",
}


def dials_folder(dataset, run, root=None):
    """Locate one cached DIALS dataset/run without a separate inventory file."""
    root = DIALS if root is None else Path(root)
    return root / dataset.strip("/").replace("/", "__") / str(run)


def cached_runs(root=None):
    """Find dataset/run folders containing Muon histograms, in stable order."""
    root = DIALS if root is None else Path(root)
    rows = []
    for folder in root.glob("*/*"):
        parts = folder.parent.name.split("__")
        if (not folder.is_dir() or not folder.name.isdigit() or int(folder.name) <= 0
                or len(parts) != 3 or not all(parts)
                or not any(path.is_file() for path in folder.glob("Muons__*.parquet"))):
            continue
        rows.append({"dataset": "/" + "/".join(parts), "run_number": int(folder.name)})
    return sorted(rows, key=lambda row: (row["dataset"], row["run_number"]))
