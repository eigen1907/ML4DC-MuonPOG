"""Plot the saved ae_base input, reconstruction, and residual for one run."""

import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
from matplotlib.ticker import MaxNLocator

from ae_base import load_run, shrink_hlt_shapes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True, help="Cached dataset/run folder")
    parser.add_argument("--group", choices=("good", "bad", "good_but_anomaly"), required=True)
    args = parser.parse_args()

    with Path("output/model/ae_base/model.pkl").open("rb") as file:
        saved = pickle.load(file)
    me_files = sorted(path.name for path in args.folder.glob("Muons__*.parquet"))
    if len(me_files) != 10:
        raise SystemExit(f"Expected 10 Muons MEs in {args.folder}")
    run = int(args.folder.name)
    dataset = "/" + args.folder.parent.name.replace("__", "/")
    data = load_run(dataset, run, me_files)
    shrink_hlt_shapes(data, saved["hlt_shape_priors"], saved["hlt_shape_prior_count"])
    features = saved["features"]
    scaled = saved["scaler"].transform(data[features].to_numpy(dtype=np.float32))
    predicted = saved["model"].predict(scaled)
    ls = data.ls_number.to_numpy()
    era = dataset.split("/")[2].split("-")[0].removeprefix("Run")
    status = "Bad" if args.group == "bad" else "Good"
    hep.style.use("CMS")

    for file in me_files:
        stem = file.removesuffix(".parquet")
        for kind, suffix in (("stat", "log_entries"), ("shape", "shape_mean")):
            index = features.index(f"{stem}__{suffix}")
            input_values = scaled[:, index]
            output_values = predicted[:, index]
            if kind == "shape":
                scale = saved["scaler"].scale_[index]
                center = saved["scaler"].mean_[index]
                input_values = input_values * scale + center
                output_values = output_values * scale + center
            units = "Stat (scaled)" if kind == "stat" else "Shape mean"
            diff = input_values - output_values
            out = Path("output/plots/ae_base") / kind / args.group / str(run) / f"{stem}.png"
            out.parent.mkdir(parents=True, exist_ok=True)
            fig, (ax, diff_ax) = plt.subplots(
                2, 1, figsize=(11, 7), sharex=True, layout="constrained",
                gridspec_kw={"height_ratios": (3, 1), "hspace": 0.05},
            )
            ax.scatter(ls, input_values, s=8, color="#1769c2", alpha=0.65, linewidths=0,
                       label="Input", rasterized=True)
            ax.scatter(ls, output_values, s=10, marker="x", color="#d65f16", alpha=0.75,
                       linewidths=0.7, label="Output", rasterized=True)
            ax.set_ylabel(units)
            if kind == "shape":
                # Keep the 0–1 reference range and show any unbounded AE outputs.
                ax.set_ylim(min(0, output_values.min() - 0.02), max(1, output_values.max() + 0.02))
            ax.legend(markerscale=4)
            hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None,
                          rlabel=f"Run {run} ({status}), {era}")
            diff_ax.scatter(ls, diff, s=8, color="#1769c2", alpha=0.75, linewidths=0,
                            rasterized=True)
            diff_ax.axhline(0, color="black", linestyle="--", linewidth=1)
            diff_ax.set(xlabel="Lumisection", ylabel="Diff")
            diff_ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
            fig.savefig(out, dpi=160, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved ae_base feature plots for run {run}")


if __name__ == "__main__":
    main()
