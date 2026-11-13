"""Shared choices for cached inputs, fixed examples and numbered experiments."""

from pathlib import Path


OUT = Path("output")
DIALS = Path("data/dials")
PROMPT_RECO_PATTERN = r"^/Muon0/Run20\d\d[A-Z]-PromptReco-v\d+/DQMIO$"
RR_DATA = Path("data/rr")
DCS_DATA = Path("data/dcs")
IDS = ["dataset", "run_number", "ls_number"]
SEED = 42
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

FEATURE_AE = "01_ae_2feat"
PILEUP_AE = "02_ae_2feat_pileup"
MDN = "04_mdn"
SHARED_GMM_AE = "05_ae_gmm_shared"
KDE_AE = "06_ae_kde"
KDE_ADAPTIVE_AE = "07_ae_kde_adaptive"
KDE_MASKED_AE = "08_ae_kde_masked"
EXPERIMENTS = {
    FEATURE_AE: "Mean/std AE trained on Good DCS-pass LS; no entry cut",
    PILEUP_AE: "Mean/std AE conditioned on OMS pile-up; pile-up is not reconstructed",
    MDN: "Pile-up-conditioned Gaussian-mixture histogram density network",
    SHARED_GMM_AE: "30 fixed Good-training Gaussians per ME, LS weights and a conditional AE",
    KDE_AE: "Binned KDE histogram inputs and a pile-up-conditioned AE",
    KDE_ADAPTIVE_AE: "Training-selected spatially adaptive KDE inputs and a conditional AE",
    KDE_MASKED_AE: "Predict each held-out ME from the other KDE histograms and pile-up",
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
