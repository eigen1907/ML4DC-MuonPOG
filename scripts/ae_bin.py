"""Train a small bin-input AE and inspect three test runs' raw histograms."""

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm, SymLogNorm
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ae_base import plot_diagnostics, plot_scores


RAW = Path("data/raw")
BASE = Path("result/ae_base")
OUT = Path("result/ae_bin")
SEED = 42
TRAIN_LS = 20_000
VALIDATION_LS = 5_000
EPOCHS = 20
THRESHOLD_QUANTILE = 0.95
CASES = {"good": 397780, "bad_well_known": 397765, "bad_ambiguous": 395713}
LABELS = {"good": "Good", "bad_well_known": "Bad (well known)",
          "bad_ambiguous": "Bad (ambiguous)"}


def load_bins(rows: pd.DataFrame, me_files: list[str], axes: dict) -> np.ndarray:
    runs = []
    for (dataset, run), group in rows.groupby(["dataset", "run_number"], sort=False):
        wanted = np.sort(group.ls_number.to_numpy(dtype=int))
        folder = RAW / dataset.strip("/").replace("/", "__") / str(run)
        mes = []
        for name in me_files:
            frame = pd.read_parquet(
                folder / name, columns=["ls_number", "data", "x_min", "x_max"],
                filters=[("ls_number", "in", wanted.tolist())],
            ).sort_values("ls_number")
            if not np.array_equal(frame.ls_number.to_numpy(), wanted):
                raise ValueError(f"Missing LS in {folder / name}")
            counts = np.stack(frame.data.to_numpy()).astype(np.float32)
            axis = (counts.shape[1], float(frame.x_min.iloc[0]), float(frame.x_max.iloc[0]))
            if (frame.x_min.nunique() != 1 or frame.x_max.nunique() != 1
                    or name in axes and axes[name] != axis):
                raise ValueError(f"Histogram axis changed: {folder / name}")
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError(f"Invalid bin contents: {folder / name}")
            axes[name] = axis
            mes.append(counts)
        runs.append(np.hstack(mes))
    return np.vstack(runs)


def plot_me(ls: np.ndarray, original: np.ndarray, predicted: np.ndarray,
            axis: tuple, label: str, era: str, run: int, me: str, output: Path) -> None:
    _, low, high = axis
    difference = original - predicted
    count_norm = LogNorm(vmin=0.1, vmax=max(1, float(original.max()), float(predicted.max())))
    delta = max(1, float(np.abs(difference).max()))
    diff_norm = SymLogNorm(linthresh=1, vmin=-delta, vmax=delta)
    short_me = me.rsplit("__", 1)[-1]
    if short_me in {"GlbMuon_Glb_eta", "GlbMuon_Glb_pt"}:
        short_me = f"{me.removeprefix('Muons__').split('__')[0]} / {short_me}"
    extent = (ls.min() - 0.5, ls.max() + 0.5, low, high)
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.png"):
        old.unlink()
    for name, values, title, norm, cmap in (
        ("original", original, "Original", count_norm, "viridis"),
        ("prediction", predicted, "AE prediction", count_norm, "viridis"),
        ("residual", difference, "Original - prediction", diff_norm, "coolwarm"),
    ):
        fig, ax = plt.subplots(figsize=(12, 6))
        fig.subplots_adjust(left=0.12, right=0.86, bottom=0.13, top=0.73)
        fig.text(0.12, 0.98, r"$\bf{CMS}$ Work in Progress", va="top")
        fig.text(0.86, 0.98, era, ha="right", va="top")
        fig.suptitle(f"{label} | Run {run} | {short_me} | {title}", y=0.87)
        image = ax.imshow(values.T, origin="lower", aspect="auto", extent=extent,
                          interpolation="nearest", norm=norm, cmap=cmap)
        ax.set(xlabel="LS", ylabel="ME x axis")
        fig.colorbar(image, ax=ax, label="Bin count" if name != "residual" else "Count difference")
        fig.savefig(output / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    metadata = pd.read_parquet(BASE / "scores.parquet").drop(columns=["anomaly_score", "flagged"])
    train = metadata[metadata.split == "train"].sample(n=TRAIN_LS, random_state=SEED)
    validation = metadata[metadata.split == "validation"].sample(n=VALIDATION_LS, random_state=SEED)
    first = train.iloc[0]
    folder = RAW / first.dataset.strip("/").replace("/", "__") / str(first.run_number)
    me_files = sorted(path.name for path in folder.glob("Muons__*.parquet"))
    if len(me_files) != 10:
        raise ValueError("Expected 10 Muons MEs")
    axes = {}
    x_train = np.log1p(load_bins(train, me_files, axes))
    x_val = np.log1p(load_bins(validation, me_files, axes))
    scaler = StandardScaler().fit(x_train)
    x_train = scaler.transform(x_train)
    x_val = scaler.transform(x_val)

    model = MLPRegressor(hidden_layer_sizes=(32, 8, 32), batch_size=512,
                         random_state=SEED, tol=0, n_iter_no_change=EPOCHS + 1)
    train_losses, val_losses = [], []
    for epoch in range(1, EPOCHS + 1):
        model.partial_fit(x_train, x_train)
        train_losses.append(float(np.mean((model.predict(x_train) - x_train) ** 2)))
        val_losses.append(float(np.mean((model.predict(x_val) - x_val) ** 2)))
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{EPOCHS}: train {train_losses[-1]:.4f}, validation {val_losses[-1]:.4f}", flush=True)

    frames, selected = [], {}
    for index, ((dataset, run), group) in enumerate(metadata.groupby(["dataset", "run_number"], sort=False), 1):
        group = group.sort_values("ls_number").copy()
        original = load_bins(group, me_files, axes)
        scaled = scaler.transform(np.log1p(original))
        reconstructed = model.predict(scaled)
        # This baseline weights every bin equally, so wider MEs contribute more.
        group["anomaly_score"] = np.mean((scaled - reconstructed) ** 2, axis=1)
        frames.append(group)
        if run in CASES.values():
            selected[run] = (group, original, reconstructed)
        if index % 100 == 0:
            print(f"Scored {index}/{metadata.run_number.nunique()} runs", flush=True)
    scores = pd.concat(frames, ignore_index=True)
    if len(scores) != len(metadata) or not np.isfinite(scores.anomaly_score).all() or (scores.anomaly_score <= 0).any():
        raise ValueError("Incomplete or invalid bin-AE scores")
    threshold = float(scores.loc[scores.split == "validation", "anomaly_score"].quantile(THRESHOLD_QUANTILE))
    scores["flagged"] = scores.anomaly_score > threshold

    OUT.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(OUT / "scores.parquet", index=False)
    with (OUT / "model.pkl").open("wb") as output:
        pickle.dump({"model": model, "scaler": scaler, "me_files": me_files, "axes": axes,
                     "threshold": threshold, "seed": SEED,
                     "train_losses": train_losses, "validation_losses": val_losses}, output)
    hep.style.use("CMS")
    plot_dir = OUT / "plots"
    plot_dir.mkdir(exist_ok=True)
    for old in plot_dir.glob("*.png"):
        old.unlink()
    for era, group in scores.groupby("era"):
        plot_scores(group, threshold, era, plot_dir / f"{era}.png")
    plot_scores(scores, threshold, "2025 all", plot_dir / "all.png")
    plot_scores(scores[scores.split == "test"], threshold, "2025 test", plot_dir / "test.png")
    plot_diagnostics(scores, train_losses, val_losses, threshold, plot_dir)

    cases = {}
    for label, run in CASES.items():
        if run not in selected:
            raise ValueError(f"Expected one complete test run: {label} {run}")
        rows, original, reconstructed = selected[run]
        if rows.label.nunique() != 1 or rows.label.iloc[0] != label or rows.split.nunique() != 1 or rows.split.iloc[0] != "test":
            raise ValueError(f"Run is not in its expected test group: {run}")
        predicted_log = scaler.inverse_transform(reconstructed)
        predicted = np.maximum(np.expm1(predicted_log), 0)
        out = OUT / "reconstructions" / f"{label}_{run}"
        offset = 0
        for name in me_files:
            bins = axes[name][0]
            plot_me(rows.ls_number.to_numpy(), original[:, offset:offset + bins],
                    predicted[:, offset:offset + bins], axes[name], LABELS[label],
                    rows.era.iloc[0], run, name.removesuffix(".parquet"),
                    out / name.removeprefix("Muons__").removesuffix(".parquet"))
            offset += bins
        cases[label] = {"run": run, "era": rows.era.iloc[0], "ls": len(rows),
                        "flagged_ls_fraction": float(scores.loc[(scores.run_number == run), "flagged"].mean()),
                        "reconstruction_mse_scaled_log_counts": float(rows.anomaly_score.mean())}
        print(f"Saved 30 ME figures for {LABELS[label]} run {run}", flush=True)
    groups = {"train_good": scores.split == "train", "validation_good": scores.split == "validation",
              "test_good": (scores.split == "test") & (scores.label == "good"),
              "test_bad_well_known": (scores.split == "test") & (scores.label == "bad_well_known"),
              "test_bad_ambiguous": (scores.split == "test") & (scores.label == "bad_ambiguous")}
    (OUT / "metrics.json").write_text(json.dumps({
        "inputs": "All stored bin contents from 10 Muons MEs, log1p then per-bin StandardScaler",
        "architecture": "1080-32-8-32-1080", "epochs": EPOCHS, "seed": SEED,
        "training_ls_sampled_from_good_runs": len(train),
        "validation_ls_sampled_from_good_runs": len(validation),
        "training_reconstruction_mse": train_losses[-1],
        "validation_reconstruction_mse": val_losses[-1],
        "threshold_quantile": THRESHOLD_QUANTILE,
        "threshold": threshold,
        "score": "Mean squared error over 1080 standardized log1p bin counts per LS",
        "flagged_ls_fraction_by_group": {name: float(scores.loc[mask, "flagged"].mean()) for name, mask in groups.items()},
        "prediction": "Inverse scaling and expm1; negative predicted counts clipped to zero for display",
        "cases": cases,
    }, indent=2) + "\n")
    lines = ["# Bin-AE plots", "", "[All LS scores](plots/all.png) · [Test LS scores](plots/test.png) · "
             "[Distributions](plots/distributions.png) · [Loss](plots/loss.png)", "",
             "Era plots: " + " · ".join(f"[{era}](plots/{era}.png)" for era in sorted(scores.era.unique())), "",
             "Each ME view uses LS on the horizontal axis and histogram position on the vertical axis.", ""]
    for label, run in CASES.items():
        lines += [f"## {LABELS[label]} — run {run}", "", "| ME | Original | AE prediction | Original - prediction |",
                  "|---|---|---|---|"]
        for name in me_files:
            stem = name.removeprefix("Muons__").removesuffix(".parquet")
            path = f"reconstructions/{label}_{run}/{stem}"
            links = [f"[{kind}]({path}/{file}.png)" for kind, file in
                     (("Original", "original"), ("Prediction", "prediction"), ("Residual", "residual"))]
            lines.append(f"| {stem.replace('__', ' / ')} | {' | '.join(links)} |")
        lines.append("")
    (OUT / "plots.md").write_text("\n".join(lines) + "\n")
    print(f"Saved {len(scores):,} bin-AE LS scores and 10 summary plots in {OUT}")


if __name__ == "__main__":
    main()
