"""Plot normalized ME heatmaps and per-LS entries distributions from cached data."""

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
    runs = frame.run_number.to_numpy(dtype=int)
    entries = frame.entries.to_numpy(dtype=np.float32)
    x_min, x_max = float(frame.iloc[0].x_min), float(frame.iloc[0].x_max)
    if ((frame.x_min != x_min).any() or (frame.x_max != x_max).any()
            or not np.isfinite([x_min, x_max]).all() or x_min >= x_max):
        raise SystemExit("Histogram ranges vary across LS; inspect before plotting")
    if not np.isfinite(counts).all() or (counts < 0).any() or not np.isfinite(entries).all() or (entries < 0).any():
        raise SystemExit("Invalid histogram contents or entries")
    totals = counts.sum(axis=1, keepdims=True)
    normalized = np.divide(counts, totals, out=np.full_like(counts, np.nan), where=totals > 0)
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
    elif short_me == "NumberOfValidRecHitsPerTrack_sta":
        axis_label = "Valid hits / track"
    if region:
        out = Path("output/plots/mes/good_but_anomaly") / region / f"{input_paths[0].stem}.png"
    else:
        out = Path("output/plots/mes") / status.lower() / str(run) / f"{input_paths[0].stem}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    hep.style.use("CMS")

    # Compress the display only: omit missing and empty LS for this ME.
    nonempty = totals[:, 0] > 0
    grid = normalized[nonempty]
    if not nonempty.any():
        grid = np.full((1, counts.shape[1]), np.nan)
    fig, ax = plt.subplots(figsize=(13.5, 8), layout="constrained")
    image = ax.imshow(
        grid.T,
        origin="lower",
        aspect="auto",
        extent=(0.5, len(grid) + 0.5, x_min, x_max),
        cmap="viridis",
        interpolation="nearest",
        vmin=0,
    )
    if short_me == "NumberOfValidRecHitsPerTrack_sta":
        ax.set_ylim(0, 65)
    if not nonempty.any():
        image.set_clim(0, 1)
        ax.text(0.5, 0.5, "No nonempty LS", transform=ax.transAxes, ha="center", va="center")
        ax.set_xticks([])
    fig.colorbar(image, ax=ax, label="Normalized bin content")
    ax.set(xlabel="Lumisection", ylabel=axis_label)
    ax.text(0.02, 0.98, short_me, transform=ax.transAxes, ha="left", va="top",
            color="white", bbox={"facecolor": "black", "alpha": 0.5, "edgecolor": "none"})
    if region:
        _, run_sizes = np.unique(runs[nonempty], return_counts=True)
        for boundary in np.cumsum(run_sizes)[:-1]:
            ax.axvline(boundary + 0.5, color="white", linewidth=0.6, alpha=0.8)
        rlabel = f"Group {region} (Good), {era}"
    else:
        rlabel = f"Run {run} ({status}), {era}"
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None,
                  rlabel=rlabel)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("Saved", out)


def plot_entries(input_paths: list[Path], region: str | None = None) -> None:
    frame = pd.concat([
        pd.read_parquet(path, columns=["dataset", "run_number", "ls_number", "me", "entries"])
        for path in input_paths
    ], ignore_index=True)
    table = frame.pivot(index=["run_number", "ls_number"], columns="me", values="entries").sort_index()
    if not np.isfinite(table.to_numpy()).all() or (table < 0).any().any():
        raise ValueError("Entries must be nonnegative and cover the same LS for all MEs")
    groups = []
    for me in table:
        values = table[me].to_numpy()
        if not any(np.array_equal(values, previous) for _, previous in groups):
            groups.append((me, values))

    run = int(frame.run_number.min())
    era = frame.dataset.iloc[0].split("/")[2].split("-")[0].removeprefix("Run")
    label = pd.read_csv("data/run_labels.csv").set_index("run_number").loc[run, "label"]
    status = "Good" if label == "good" else "Bad"
    relative = Path("good_but_anomaly") / region if region else Path(status.lower()) / str(run)
    out = Path("output/plots/mes") / relative / "entries.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    rlabel = f"Group {region}" if region else f"Run {run}"
    rlabel += f" ({status}), {era}"
    names = {
        "relPFIso_R03": "GlbMuon",
        "GlbMuon_Glb_eta": "GlbMuon",
        "GlbMuon_Glb_phi": "GlbMuon",
        "GlbMuon_Glb_pt": "GlbMuon",
        "GlbGlbMuon_LM": "GlbGlbMuonLM",
        "GlbGlbMuon_HM": "GlbGlbMuonHM",
        "NumberOfMeanRecHitsPerTrack_glb": "GlbMuonHits",
        "NumberOfValidRecHitsPerTrack_sta": "StaMuonHits",
    }
    maximum = max(1, max(values.max() for _, values in groups))
    bins = np.unique(np.r_[-0.5, np.rint(np.geomspace(1, maximum + 1, 60)) - 0.5])
    hep.style.use("CMS")
    fig, ax = plt.subplots(figsize=(13.5, 8), layout="constrained")
    largest = 0
    for me, values in groups:
        short_me = me.rsplit("/", 1)[-1]
        name = "GlbMuonOneHLT" if "/MuonRecoOneHLT/" in me else names.get(short_me, short_me)
        counts, _, _ = ax.hist(values, bins=bins, histtype="step", linewidth=2,
                               label=rf"{name} ($\mu$ = {values.mean():,.1f})")
        largest = max(largest, counts.max())
    ax.set_xscale("symlog", linthresh=1)
    ax.set(xlabel="Entries", ylabel="LS", xlim=(bins[0], 1e6), ylim=(0, largest * 1.8))
    ax.legend(loc="upper left")
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel=rlabel)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}: {len(groups)} unique entries profiles, {len(table)} LS (including zeros)")


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
        plot_entries(args.input, args.region)
    else:
        groups = defaultdict(list)
        for input_path in args.input:
            plot_me([input_path])
            groups[input_path.parent].append(input_path)
        for input_paths in groups.values():
            plot_entries(input_paths)


if __name__ == "__main__":
    main()
