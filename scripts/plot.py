"""Plot raw per-LS histogram contents from cached 1D MEs."""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd


def plot_me(input_paths: list[Path], region: str | None = None) -> None:
    frames = [pd.read_parquet(path) for path in input_paths]
    if any(frame.empty or frame.me.nunique() != 1 or frame.run_number.nunique() != 1
           or frame.dataset.nunique() != 1 for frame in frames):
        raise SystemExit("Expected one nonempty ME and one run in each cached file")
    frame = pd.concat(frames).sort_values(["run_number", "ls_number"])
    if frame.me.nunique() != 1 or frame.duplicated(["run_number", "ls_number"]).any():
        raise SystemExit("Expected one ME and unique run/LS pairs")
    if frame.groupby("run_number").dataset.nunique().gt(1).any():
        raise SystemExit("Multiple datasets found for one run")
    if frame["data"].map(len).nunique() != 1:
        raise SystemExit("Histogram bin counts vary across runs")
    counts = np.stack(frame["data"].to_numpy()).astype(np.float32)
    ls = frame.ls_number.to_numpy(dtype=int)
    runs = frame.run_number.to_numpy(dtype=int)
    entries = frame.entries.to_numpy(dtype=np.float32)
    x_min, x_max = float(frame.iloc[0].x_min), float(frame.iloc[0].x_max)
    if ((frame.x_min != x_min).any() or (frame.x_max != x_max).any()
            or not np.isfinite([x_min, x_max]).all() or x_min >= x_max):
        raise SystemExit("Histogram ranges vary across LS; inspect before plotting")
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.isfinite(entries).all() or (entries < 0).any():
        raise SystemExit("Invalid histogram contents or entries")
    run = int(runs[0])
    short_me = str(frame.iloc[0].me).rsplit("/", 1)[-1]
    eras = frame.dataset.map(lambda dataset: str(dataset).split("/")[2].split("-")[0].removeprefix("Run")).unique()
    if len(eras) != 1:
        raise SystemExit("Expected one era in each plot")
    era = eras[0]
    labels = pd.read_csv("data/run_labels.csv").set_index("run_number").label
    statuses = [{"good": "Good", "bad_well_known": "Bad", "bad_ambiguous": "Bad"}.get(labels.get(number))
                for number in np.unique(runs)]
    if None in statuses:
        raise SystemExit("No Good/Bad label for every run")
    if region and any(status != "Good" for status in statuses):
        raise SystemExit("Region plots require Good runs only")
    status = statuses[0]
    axis_label = "ME x axis"
    if short_me.endswith("_eta"):
        axis_label = "Muon $\\eta$"
    elif short_me.endswith("_pt"):
        axis_label = "Muon $p_T$"
    elif short_me.endswith("_phi"):
        axis_label = "Muon $\\phi$"
    if region:
        out = Path("output/plots/mes/good_but_anomaly") / region / f"{input_paths[0].stem}.png"
    else:
        out = Path("output/plots/mes") / status.lower() / str(run) / f"{input_paths[0].stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    hep.style.use("CMS")

    if region:
        grid = counts
        x_min_plot, x_max_plot = -0.5, len(frame) - 0.5
    else:
        grid = np.full((ls.max() - ls.min() + 1, counts.shape[1]), np.nan)
        grid[ls - ls.min()] = counts
        x_min_plot, x_max_plot = ls.min() - 0.5, ls.max() + 0.5
    fig, ax = plt.subplots(figsize=(13.5, 8), layout="constrained")
    image = ax.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        extent=(x_min_plot, x_max_plot, x_min, x_max),
        cmap="viridis",
    )
    if not counts.any():
        image.set_clim(0, 1)
    fig.colorbar(image, ax=ax, label="Bin content")
    if region:
        run_numbers, run_sizes = np.unique(runs, return_counts=True)
        ends = np.cumsum(run_sizes)
        midpoints = (np.r_[0, ends[:-1]] + ends - 1) / 2
        targets = np.linspace(0, len(frame) - 1, min(4, len(run_numbers)))
        tick_ids = np.unique(np.argmin(np.abs(midpoints[:, None] - targets), axis=0))
        ax.set_xticks(midpoints[tick_ids], run_numbers[tick_ids])
        for boundary in ends[:-1]:
            ax.axvline(boundary - 0.5, color="white", linewidth=0.6, alpha=0.8)
        ax.set(xlabel="Run number", ylabel=axis_label)
        rlabel = f"Group {region} (Good), {era}"
    else:
        ax.set(xlabel="Lumisection", ylabel=axis_label)
        rlabel = f"Run {run} ({status}), {era}"
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None,
                  rlabel=rlabel)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Parquet file(s) from fetch.py")
    parser.add_argument("--region", choices="ABCD")
    args = parser.parse_args()

    if args.region:
        groups = defaultdict(list)
        for input_path in args.input:
            groups[input_path.stem].append(input_path)
        for input_paths in groups.values():
            plot_me(input_paths, args.region)
    else:
        for input_path in args.input:
            plot_me([input_path])


if __name__ == "__main__":
    main()
