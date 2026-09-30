"""Make three CMS-style diagnostic figures from one cached 1D ME."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Parquet file from fetch.py")
    args = parser.parse_args()

    frame = pd.read_parquet(args.input).sort_values("ls_number")
    if frame.empty or frame.me.nunique() != 1 or frame.run_number.nunique() != 1:
        raise SystemExit("Expected one nonempty ME and one run in the cached file")
    counts = np.stack(frame["data"].to_numpy()).astype(float)
    ls = frame.ls_number.to_numpy(dtype=int)
    x_min, x_max = float(frame.iloc[0].x_min), float(frame.iloc[0].x_max)
    if (frame.x_min != x_min).any() or (frame.x_max != x_max).any():
        raise SystemExit("Histogram ranges vary across LS; inspect before plotting")
    edges = np.linspace(x_min, x_max, counts.shape[1] + 1)
    centers = (edges[:-1] + edges[1:]) / 2
    run = int(frame.iloc[0].run_number)
    me = str(frame.iloc[0].me)
    short_me = me.rsplit("/", 1)[-1]
    axis_label = "ME x axis"
    if short_me.endswith("_eta"):
        axis_label = "Muon $\\eta$"
    elif short_me.endswith("_pt"):
        axis_label = "Muon $p_T$"
    elif short_me.endswith("_phi"):
        axis_label = "Muon $\\phi$"
    out = Path("plots") / args.input.relative_to("data/raw").with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    hep.style.use("CMS")

    grid = np.full((ls.max() - ls.min() + 1, counts.shape[1]), np.nan)
    grid[ls - ls.min()] = counts
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.13, top=0.78)
    image = ax.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        extent=(ls.min() - 0.5, ls.max() + 0.5, x_min, x_max),
        cmap="viridis",
    )
    fig.colorbar(image, ax=ax, label="Bin content")
    ax.set(xlabel="Lumisection", ylabel=axis_label)
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel=f"Run {run} | {short_me}")
    fig.savefig(out / "contents.png", dpi=160)
    plt.close(fig)

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.10, top=0.78, hspace=0.10)
    totals = counts.sum(axis=1)
    mean_x = np.divide(
        counts @ centers,
        totals,
        out=np.full(len(totals), np.nan),
        where=totals > 0,
    )
    top.plot(ls, totals, marker=".", linestyle="none")
    bottom.plot(ls, mean_x, marker=".", linestyle="none")
    top.set(ylabel="Sum of bin contents")
    bottom.set(xlabel="Lumisection", ylabel=f"Mean {axis_label}")
    hep.cms.label(ax=top, text="Work in Progress", data=True, com=None, rlabel=f"Run {run} | {short_me}")
    fig.savefig(out / "statistics.png", dpi=160)
    plt.close(fig)

    nonzero = np.flatnonzero(totals > 0)
    selected = sorted({int(nonzero[0]), int(nonzero[len(nonzero) // 2]), int(nonzero[-1])}) if len(nonzero) else [0]
    fig, ax = plt.subplots(figsize=(15, 8))
    fig.subplots_adjust(left=0.13, right=0.96, bottom=0.13, top=0.78)
    for index in selected:
        ax.stairs(counts[index], edges, label=f"LS {ls[index]}")
    ax.set(xlabel=axis_label, ylabel="Bin content")
    ax.legend(frameon=False)
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel=f"Run {run} | {short_me}")
    fig.savefig(out / "examples.png", dpi=160)
    plt.close(fig)
    print("Saved three figures in", out)


if __name__ == "__main__":
    main()
