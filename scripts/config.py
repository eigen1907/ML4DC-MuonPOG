"""Shared study choices: cached inputs, fixed examples and five experiments."""

from pathlib import Path


OUT = Path("output")
DIALS = OUT / "dataset/dials"
PROMPT_RECO_PATTERN = r"/Run2025[A-Z]-PromptReco-v\d+/DQMIO$"
RR_DATA = Path("data/rr")
DCS_PATH = Path("data/dcs/2025.json")
IDS = ["dataset", "run_number", "ls_number"]
SEED = 42
WINDOW_SIZE = 5
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

BASE = "01_ae_2feat"
BASE_DCS = "02_ae_2feat_dcs"
HIST_DCS = "03_ae_hist_dcs"
DM_DCS = "04_ae_dm_k1_dcs"
DM_WINDOW = "05_ae_dm_k1_weighted_window"
EXPERIMENTS = {
    BASE: "Frozen historical mean/std AE; original unfiltered Good training",
    BASE_DCS: "Same mean/std AE trained on Good DCS-pass LS; no entry cut",
    HIST_DCS: "Native-histogram AE trained and scored with MSE; Good DCS-pass LS",
    DM_DCS: "K=1 native-histogram DM AE; variance-weighted reconstruction score",
    DM_WINDOW: "Same DM checkpoint; centered LS-window mean of its weighted score",
}


def dials_folder(dataset, run, root=None):
    """Locate one cached DIALS dataset/run without a separate inventory file."""
    root = DIALS if root is None else root
    return root / dataset.strip("/").replace("/", "__") / str(run)


def cached_runs():
    """Find dataset/run folders containing Muon histograms, in stable order."""
    rows = []
    for folder in DIALS.glob("*/*"):
        parts = folder.parent.name.split("__")
        if (not folder.is_dir() or not folder.name.isdigit() or int(folder.name) <= 0
                or len(parts) != 3 or not all(parts)
                or not any(path.is_file() for path in folder.glob("Muons__*.parquet"))):
            continue
        rows.append({"dataset": "/" + "/".join(parts), "run_number": int(folder.name)})
    return sorted(rows, key=lambda row: (row["dataset"], row["run_number"]))
