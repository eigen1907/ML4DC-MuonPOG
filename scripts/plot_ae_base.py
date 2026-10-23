"""Plot ae_base mean/std reconstructions and diagnostic entries for a run or region."""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import mplhep as hep
import numpy as np
import pandas as pd

from ae_base import load_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, nargs="+", required=True, help="Cached dataset/run folder(s)")
    parser.add_argument("--group", choices=("good", "bad", "A", "B", "C", "D"), required=True)
    args = parser.parse_args()

    with Path("output/model/ae_base/model.pkl").open("rb") as file:
        saved = pickle.load(file)
    region = args.group in ("A", "B", "C", "D")
    if not region and len(args.folder) != 1:
        raise SystemExit("Individual Good/Bad plots require one run folder")
    me_files = sorted(path.name for path in args.folder[0].glob("Muons__*.parquet"))
    if len(me_files) != 10:
        raise SystemExit(f"Expected 10 Muons MEs in {args.folder[0]}")
    frames, eras = [], set()
    for folder in args.folder:
        if sorted(path.name for path in folder.glob("Muons__*.parquet")) != me_files:
            raise SystemExit(f"ME list differs: {folder}")
        dataset = "/" + folder.parent.name.replace("__", "/")
        frames.append(load_run(dataset, int(folder.name), me_files))
        eras.add(dataset.split("/")[2].split("-")[0].removeprefix("Run"))
    data = pd.concat(frames, ignore_index=True).sort_values(["run_number", "ls_number"]).reset_index(drop=True)
    if len(eras) != 1 or data.duplicated(["run_number", "ls_number"]).any():
        raise SystemExit("Expected one era and unique (run, LS) rows")
    status = "Bad" if args.group == "bad" else "Good"
    labels = pd.read_csv("data/run_labels.csv").set_index("run_number").label
    run_status = labels.reindex(data.run_number.unique()).map(
        {"good": "Good", "bad_well_known": "Bad", "bad_ambiguous": "Bad"})
    if not run_status.eq(status).all():
        raise SystemExit("Run labels disagree with the requested plot group")
    features = saved["features"]
    scaled = saved["scaler"].transform(data[features].to_numpy(dtype=np.float32))
    predicted = saved["model"].predict(scaled)
    x = np.arange(len(data)) if region else data.ls_number.to_numpy()
    run = int(data.run_number.iloc[0])
    relative = Path("good_but_anomaly") / args.group if region else Path(args.group) / str(run)
    rlabel = f"Group {args.group}" if region else f"Run {run}"
    rlabel += f" ({status}), {next(iter(eras))}"
    if region:
        sizes = data.groupby("run_number", sort=True).size()
        edges = np.r_[0, sizes.to_numpy().cumsum()]
        midpoints = (edges[:-1] + edges[1:] - 1) / 2
        ticks = np.unique(np.abs(midpoints[:, None] - np.linspace(0, len(data) - 1, 4)).argmin(axis=0))
    hep.style.use("CMS")

    for file in me_files:
        stem = file.removesuffix(".parquet")
        for kind in ("stat", "mean", "std"):
            if kind != "stat" and f"{stem}__{kind}" not in features:
                continue
            out = Path("output/plots/ae_base") / kind / relative / f"{stem}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            if kind == "stat":
                fig, ax = plt.subplots(figsize=(11, 6), layout="constrained")
                ax.scatter(x, data[f"{stem}__entries"], s=8, color="#1769c2", alpha=0.65,
                           linewidths=0, rasterized=True)
                ax.set_ylabel("Entries")
                ax.set_ylim(bottom=0)
                bottom_ax = ax
                axes = (ax,)
            else:
                index = features.index(f"{stem}__{kind}")
                scale = saved["scaler"].scale_[index]
                center = saved["scaler"].mean_[index]
                input_values = scaled[:, index] * scale + center
                output_values = predicted[:, index] * scale + center
                fig, (ax, diff_ax) = plt.subplots(
                    2, 1, figsize=(11, 7), sharex=True, layout="constrained",
                    gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05},
                )
                ax.scatter(x, input_values, s=8, color="#1769c2", alpha=0.65, linewidths=0,
                           label="Input", rasterized=True)
                ax.scatter(x, output_values, s=10, marker="x", color="#d65f16", alpha=0.75,
                           linewidths=0.7, label="Output", rasterized=True)
                ax.set_ylabel(kind.capitalize())
                # Show the physical reference range and any unbounded AE outputs.
                upper = 1 if kind == "mean" else 0.5
                ax.set_ylim(min(0, output_values.min() - 0.02), max(upper, output_values.max() + 0.02))
                ax.add_artist(AnchoredText(stem.rsplit("__", 1)[-1], loc="upper left", frameon=False))
                ax.legend(loc="upper right", markerscale=4)
                diff = input_values - output_values
                diff_ax.scatter(x, diff, s=8, color="#7b3294", alpha=0.75,
                                linewidths=0, rasterized=True)
                diff_ax.axhline(0, color="black", linestyle="--", linewidth=1)
                limit = 0.1 if kind == "mean" else 0.02
                diff_ax.set(ylabel="Diff", ylim=(-limit, limit), yticks=(-limit, 0, limit))
                for mask, marker, boundary in ((diff > limit, "^", limit), (diff < -limit, "v", -limit)):
                    diff_ax.scatter(x[mask], np.full(mask.sum(), 0.95 * boundary), s=22,
                                    marker=marker, color="#7b3294", linewidths=0, rasterized=True)
                bottom_ax = diff_ax
                axes = (ax, diff_ax)
            hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None,
                          rlabel=rlabel)
            bottom_ax.set_xlabel("Run number" if region else "Lumisection")
            if region:
                bottom_ax.set_xlim(-0.5, len(data) - 0.5)
                bottom_ax.set_xticks(midpoints[ticks], sizes.index.to_numpy()[ticks].astype(str))
                for boundary in edges[1:-1] - 0.5:
                    for panel in axes:
                        panel.axvline(boundary, color="gray", linewidth=0.5, alpha=0.4, zorder=0)
            fig.savefig(out, dpi=160, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved ae_base feature plots for {rlabel}: {len(data)} LS")


if __name__ == "__main__":
    main()
