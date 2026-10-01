"""Fit a small autoencoder on provisional-good 2025 Muon0 runs and score each LS."""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


RAW = Path("data/raw")
OUT = Path("result/ae_base")
SEED = 42
THRESHOLD_QUANTILE = 0.95
EPOCHS = 80
ETA_ME = "Muons__MuonRecoAnalyzer__GlbMuon_Glb_eta.parquet"


def load_run(dataset: str, run: int, me_files: list[str]) -> pd.DataFrame:
    folder = RAW / dataset.strip("/").replace("/", "__") / str(run)
    result = None
    for name in me_files:
        frame = pd.read_parquet(
            folder / name, columns=["ls_number", "entries", "data", "x_min", "x_max"]
        ).sort_values("ls_number")
        ls = frame.ls_number.to_numpy()
        if len(ls) == 0 or len(np.unique(ls)) != len(ls):
            raise ValueError(f"Empty or duplicate LS: {folder / name}")
        if result is None:
            result = pd.DataFrame({"dataset": dataset, "run_number": run, "ls_number": ls})
        elif not np.array_equal(ls, result.ls_number.to_numpy()):
            raise ValueError(f"ME LS coverage differs: {folder / name}")
        entries = frame.entries.to_numpy(dtype=np.float32)
        if not np.isfinite(entries).all() or (entries < 0).any():
            raise ValueError(f"Invalid entries: {folder / name}")
        counts = np.stack(frame.data.to_numpy()).astype(np.float32)
        if not np.isfinite(counts).all() or (counts < 0).any():
            raise ValueError(f"Invalid histogram contents: {folder / name}")
        if frame.x_min.nunique() != 1 or frame.x_max.nunique() != 1:
            raise ValueError(f"Axis changes within run: {folder / name}")
        low, high = float(frame.x_min.iloc[0]), float(frame.x_max.iloc[0])
        if not np.isfinite([low, high]).all() or low >= high:
            raise ValueError(f"Invalid axis: {folder / name}")
        edges = np.linspace(low, high, counts.shape[1] + 1, dtype=np.float32)
        centers = (edges[:-1] + edges[1:]) / 2
        positions = np.abs(centers) / max(abs(low), abs(high)) if name.endswith("__GlbMuon_Glb_eta.parquet") else (centers - low) / (high - low)
        total = counts.sum(axis=1)
        shape_mean = np.divide(
            counts @ positions, total, out=np.zeros(len(total), dtype=np.float32), where=total > 0
        )
        stem = name.removesuffix(".parquet")
        result[f"{stem}__log_entries"] = np.log1p(entries)
        result[f"{stem}__shape_mean"] = shape_mean
        if name == ETA_ME:
            result["eta_entries"] = entries
    return result


def plot_scores(frame: pd.DataFrame, threshold: float, era: str, output: Path) -> None:
    backgrounds = {"good": "#cde8d1", "bad_well_known": "#f3bcbc", "bad_ambiguous": "#ffe599"}
    fig, ax = plt.subplots(figsize=(16, 6), layout="constrained")
    frame = frame.sort_values(["run_number", "ls_number"]).reset_index(drop=True)
    runs, midpoints = [], []
    for run, group in frame.groupby("run_number", sort=False):
        first, last = int(group.index[0]), int(group.index[-1])
        ax.axvspan(first - 0.5, last + 0.5, color=backgrounds[group.label.iloc[0]], zorder=0)
        runs.append(run)
        midpoints.append((first + last) / 2)
    ax.scatter(np.arange(len(frame)), frame.anomaly_score, s=9, alpha=0.5,
               color="#1769c2", linewidths=0, rasterized=True)
    ax.axhline(threshold, color="black", linestyle="--")
    ax.set_yscale("log")
    ax.set(xlabel="Run number", ylabel="Anomaly score", xlim=(-0.5, len(frame) - 0.5))
    ticks = {0, len(runs) - 1}
    ticks.update(np.argmin(np.abs(np.asarray(midpoints) - target))
                 for target in np.linspace(0, len(frame) - 1, 8))
    ax.set_xticks([midpoints[i] for i in sorted(ticks)], [str(runs[i]) for i in sorted(ticks)])
    ax.legend(handles=[
        Patch(facecolor=backgrounds["good"], edgecolor="#5a9e68", label="Good"),
        Patch(facecolor=backgrounds["bad_well_known"], edgecolor="#be6868", label="Bad (well known)"),
        Patch(facecolor=backgrounds["bad_ambiguous"], edgecolor="#b89332", label="Bad (ambiguous)"),
        Line2D([], [], color="black", linestyle="--", label="Threshold"),
        Line2D([], [], marker="o", linestyle="none", color="#1769c2", label="LS"),
    ], ncol=3, frameon=True, facecolor="white", framealpha=0.95)
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel=era)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_diagnostics(scores: pd.DataFrame, train_losses: list[float], val_losses: list[float],
                     threshold: float, plot_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train")
    ax.plot(range(1, len(val_losses) + 1), val_losses, label="Validation")
    ax.set(xlabel="Epoch", ylabel="Reconstruction MSE")
    ax.legend()
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel="2025")
    fig.savefig(plot_dir / "loss.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    groups = [("good", "Good", "#27834b"),
              ("bad_well_known", "Bad (well known)", "#c43c3c"),
              ("bad_ambiguous", "Bad (ambiguous)", "#ae7900")]
    selected = scores[scores.split == "test"]
    bins = np.geomspace(selected.anomaly_score.min(), selected.anomaly_score.max(), 60)
    for name, label, color in groups:
        values = selected.loc[selected.label == name, "anomaly_score"]
        ax.hist(values, bins=bins, weights=np.full(len(values), 1 / len(values)),
                histtype="step", linewidth=2, color=color, label=f"{label} ({len(values):,} LS)")
    ax.axvline(threshold, color="black", linestyle="--", label="Threshold")
    ax.set(xscale="log", xlabel="Anomaly score", ylabel="Fraction of LS / bin")
    ax.legend()
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel="2025 test")
    fig.savefig(plot_dir / "distributions.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    manifest = pd.read_csv("data/runs.csv")
    manifest = manifest[manifest.dataset.str.contains(r"/Run2025[A-Z]-PromptReco-v\d+/DQMIO$")].copy()
    labels = pd.read_csv("data/run_labels.csv")
    allowed = {"good", "bad_well_known", "bad_ambiguous"}
    listed = set(labels.run_number)
    bad = set(labels.loc[labels.label == "bad_well_known", "run_number"])
    ambiguous = set(labels.loc[labels.label == "bad_ambiguous", "run_number"])
    runs = set(manifest.run_number)
    if (len(manifest) != len(runs) or len(labels) != len(listed) or runs != listed
            or not labels.label.isin(allowed).all() or not bad or not ambiguous):
        raise SystemExit("PromptReco runs and data/run_labels.csv do not agree")
    manifest["era"] = manifest.dataset.str.extract(r"Run(2025[A-Z])")[0]
    good = manifest[~manifest.run_number.isin(bad | ambiguous)]
    train, holdout = train_test_split(good, train_size=0.70, random_state=SEED, stratify=good.era)
    validation, test = train_test_split(holdout, train_size=0.5, random_state=SEED, stratify=holdout.era)
    split = {"train": set(train.run_number), "validation": set(validation.run_number),
             "test": set(test.run_number) | bad | ambiguous}

    expected_runs = len(manifest)
    first = manifest.iloc[0]
    first_folder = RAW / first.dataset.strip("/").replace("/", "__") / str(first.run_number)
    me_files = sorted(path.name for path in first_folder.glob("Muons__*.parquet"))
    if len(me_files) != 10 or ETA_ME not in me_files:
        raise SystemExit("Expected all 10 cached Muons/ MEs in the first PromptReco run")
    ready = []
    for row in manifest.itertuples():
        folder = RAW / row.dataset.strip("/").replace("/", "__") / str(row.run_number)
        ready.append(all((folder / name).exists() for name in me_files))
    manifest = manifest.loc[ready].copy()
    if not len(manifest):
        raise SystemExit("No complete PromptReco runs in the raw cache")
    print(f"Using {len(manifest)}/{expected_runs} complete PromptReco runs from the current cache", flush=True)

    frames = []
    for index, row in enumerate(manifest.itertuples(), 1):
        frame = load_run(row.dataset, row.run_number, me_files)
        if len(frame) != row.ls_count:
            raise ValueError(f"DIALS LS count differs for run {row.run_number}: {len(frame)} vs {row.ls_count}")
        frame["era"] = row.era
        frame["label"] = "bad_ambiguous" if row.run_number in ambiguous else "bad_well_known" if row.run_number in bad else "good"
        frame["split"] = next(name for name, members in split.items() if row.run_number in members)
        frames.append(frame)
        if index % 100 == 0:
            print(f"Loaded {index}/{len(manifest)} runs", flush=True)
    data = pd.concat(frames, ignore_index=True)
    features = [f"{name.removesuffix('.parquet')}__{kind}" for name in me_files for kind in ("log_entries", "shape_mean")]
    scaler = StandardScaler().fit(data.loc[data.split == "train", features].to_numpy(dtype=np.float32))
    x_train = scaler.transform(data.loc[data.split == "train", features].to_numpy(dtype=np.float32))
    x_val = scaler.transform(data.loc[data.split == "validation", features].to_numpy(dtype=np.float32))
    model = MLPRegressor(hidden_layer_sizes=(8, 3, 8), batch_size=1024, random_state=SEED,
                         tol=0, n_iter_no_change=EPOCHS + 1)
    train_losses, val_losses = [], []
    for epoch in range(1, EPOCHS + 1):
        model.partial_fit(x_train, x_train)
        train_losses.append(float(np.mean((model.predict(x_train) - x_train) ** 2)))
        val_losses.append(float(np.mean((model.predict(x_val) - x_val) ** 2)))
        if epoch % 20 == 0:
            print(f"Epoch {epoch}/{EPOCHS}: train {train_losses[-1]:.4f}, validation {val_losses[-1]:.4f}", flush=True)
    x = scaler.transform(data[features].to_numpy(dtype=np.float32))
    data["anomaly_score"] = np.mean((model.predict(x) - x) ** 2, axis=1)
    threshold = float(data.loc[data.split == "validation", "anomaly_score"].quantile(THRESHOLD_QUANTILE))
    data["flagged"] = data.anomaly_score > threshold

    OUT.mkdir(parents=True, exist_ok=True)
    scores = data[["dataset", "run_number", "ls_number", "era", "label", "split", "eta_entries", "anomaly_score", "flagged"]]
    scores.to_parquet(OUT / "scores.parquet", index=False)
    with (OUT / "model.pkl").open("wb") as output:
        pickle.dump({"model": model, "scaler": scaler, "threshold": threshold,
                     "features": features, "seed": SEED,
                     "train_losses": train_losses, "validation_losses": val_losses,
                     "run_split": {key: sorted(value & set(manifest.run_number)) for key, value in split.items()}}, output)
    groups = {"train_good": scores.split == "train", "validation_good": scores.split == "validation",
              "test_good": (scores.split == "test") & (scores.label == "good"),
              "test_bad_well_known": (scores.split == "test") & (scores.label == "bad_well_known"),
              "test_bad_ambiguous": (scores.split == "test") & (scores.label == "bad_ambiguous")}
    metrics = {
        "labels": "run groups are applied to LS for visualization; no LS-level ground truth",
        "split": "provisional-good runs, stratified by era: 70% train / 15% validation / 15% test; both bad groups test-only",
        "threshold_quantile": THRESHOLD_QUANTILE,
        "cache_snapshot_promptreco_runs": len(manifest), "catalog_promptreco_runs": expected_runs,
        "runs_with_ls_completeness_below_one": int((manifest.ls_completeness < 1).sum()),
        "feature_count": len(features), "training_epochs": EPOCHS,
        "training_reconstruction_mse": train_losses[-1],
        "validation_reconstruction_mse": val_losses[-1],
        "threshold": threshold,
        "run_counts": {name: int(scores.loc[mask, "run_number"].nunique()) for name, mask in groups.items()},
        "ls_counts": {name: int(mask.sum()) for name, mask in groups.items()},
        "flagged_ls_fraction_by_group": {name: float(scores.loc[mask, "flagged"].mean()) for name, mask in groups.items()},
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    hep.style.use("CMS")
    plot_dir = OUT / "plots"
    plot_dir.mkdir(exist_ok=True)
    for old_plot in plot_dir.glob("*.png"):
        old_plot.unlink()
    for era, frame in scores.groupby("era"):
        plot_scores(frame, threshold, era, plot_dir / f"{era}.png")
    plot_scores(scores, threshold, "2025 all", plot_dir / "all.png")
    test_scores = scores[scores.split == "test"]
    plot_scores(test_scores, threshold, "2025 test", plot_dir / "test.png")
    plot_diagnostics(scores, train_losses, val_losses, threshold, plot_dir)
    print(f"Saved {len(scores)} LS scores, metrics, model, and {scores.era.nunique() + 4} plots in {OUT}")
    print(f"Provisional-good test LS flag fraction: {metrics['flagged_ls_fraction_by_group']['test_good']:.3%}")


if __name__ == "__main__":
    main()
