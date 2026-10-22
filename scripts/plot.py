"""Plot LS-by-bin contents from one cached 1D ME."""

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
    counts = np.stack(frame["data"].to_numpy()).astype(float)
    ls = frame.ls_number.to_numpy(dtype=int)
    x_min, x_max = float(frame.iloc[0].x_min), float(frame.iloc[0].x_max)
    if (frame.x_min != x_min).any() or (frame.x_max != x_max).any():
        raise SystemExit("Histogram ranges vary across LS; inspect before plotting")
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
    out = Path("output/plots/mes") / group / str(run) / f"{args.input.stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
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
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


if __name__ == "__main__":
    main()
