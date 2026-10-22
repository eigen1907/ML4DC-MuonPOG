"""Plot raw contents and the two ae_base inputs from one cached 1D ME."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Parquet file from fetch.py")
    parser.add_argument("--good-but-anomaly", action="store_true")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input).sort_values("ls_number")
    if frame.empty or frame.me.nunique() != 1 or frame.run_number.nunique() != 1 or frame.dataset.nunique() != 1:
        raise SystemExit("Expected one nonempty ME and one run in the cached file")
    counts = np.stack(frame["data"].to_numpy()).astype(np.float32)
    ls = frame.ls_number.to_numpy(dtype=int)
    entries = frame.entries.to_numpy(dtype=np.float32)
    x_min, x_max = float(frame.iloc[0].x_min), float(frame.iloc[0].x_max)
    if ((frame.x_min != x_min).any() or (frame.x_max != x_max).any()
            or not np.isfinite([x_min, x_max]).all() or x_min >= x_max):
        raise SystemExit("Histogram ranges vary across LS; inspect before plotting")
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.isfinite(entries).all() or (entries < 0).any():
        raise SystemExit("Invalid histogram contents or entries")
    run = int(frame.iloc[0].run_number)
    short_me = str(frame.iloc[0].me).rsplit("/", 1)[-1]
    era = str(frame.iloc[0].dataset).split("/")[2].split("-")[0].removeprefix("Run")
    label = pd.read_csv("data/run_labels.csv").set_index("run_number").label.get(run)
    status = {"good": "Good", "bad_well_known": "Bad", "bad_ambiguous": "Bad"}.get(label)
    if status is None:
        raise SystemExit(f"No Good/Bad label for run {run}")
    if args.good_but_anomaly and status != "Good":
        raise SystemExit(f"Run {run} is not labeled Good")
    axis_label = "ME x axis"
    if short_me.endswith("_eta"):
        axis_label = "Muon $\\eta$"
    elif short_me.endswith("_pt"):
        axis_label = "Muon $p_T$"
    elif short_me.endswith("_phi"):
        axis_label = "Muon $\\phi$"
    group = "good_but_anomaly" if args.good_but_anomaly else status.lower()
    out = Path("output/plots/mes") / group / str(run) / args.input.stem
    out.mkdir(parents=True, exist_ok=True)
    hep.style.use("CMS")

    grid = np.full((ls.max() - ls.min() + 1, counts.shape[1]), np.nan)
    grid[ls - ls.min()] = counts
    fig, ax = plt.subplots(figsize=(13.5, 8), layout="constrained")
    image = ax.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        extent=(ls.min() - 0.5, ls.max() + 0.5, x_min, x_max),
        cmap="viridis",
    )
    fig.colorbar(image, ax=ax, label="Bin content")
    ax.set(xlabel="Lumisection", ylabel=axis_label)
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None,
                  rlabel=f"Run {run} ({status}), {era}")
    fig.savefig(out / "me.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # Match ae_base's shape_mean and log_entries inputs for this ME.
    edges = np.linspace(x_min, x_max, counts.shape[1] + 1, dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) / 2
    positions = (np.abs(centers) / max(abs(x_min), abs(x_max))
                 if short_me.endswith("_eta") else (centers - x_min) / (x_max - x_min))
    total = counts.sum(axis=1)
    shape = np.divide(counts @ positions, total, out=np.zeros(len(total), dtype=np.float32), where=total > 0)
    for name, feature, ylabel in (
        ("shape", shape, "Shape mean"),
        ("stat", np.log1p(entries), "log1p(entries)"),
    ):
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        ax.scatter(ls, feature, s=8, alpha=0.65, color="#1769c2", linewidths=0, rasterized=True)
        ax.set(xlabel="Lumisection", ylabel=ylabel, xlim=(ls.min() - 0.5, ls.max() + 0.5))
        if name == "shape":
            ax.set_ylim(0, 1)
        hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None,
                      rlabel=f"Run {run} ({status}), {era}")
        fig.savefig(out / f"{name}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
