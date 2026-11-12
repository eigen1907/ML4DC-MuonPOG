"""Plot saved LS scores and optional training losses; no model fitting."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import mplhep as hep
import numpy as np
import pandas as pd

from scripts.models.config import IDS, OUT, REGIONS
from scripts.data.dials import load_run_labels


BLUE, GREEN, RED, GREY = "#1769c2", "#cde8d1", "#f3bcbc", "#e0e0e0"


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _ordered(scores):
    if scores.duplicated(IDS).any():
        raise ValueError("Duplicate dataset/run/LS rows")
    return scores.loc[scores.dcs_pass.fillna(False)].sort_values(
        ["run_number", "ls_number", "dataset"]).reset_index(drop=True)


def _values(frame, threshold):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("A positive finite threshold is required")
    values = frame.anomaly_score.to_numpy(dtype=float, na_value=np.nan) / threshold
    if (values[np.isfinite(values)] < 0).any():
        raise ValueError("Anomaly scores must be nonnegative")
    return values


def _limits(values):
    positive = np.r_[1., values[np.isfinite(values) & (values > 0)]]
    return positive.min() * .8, positive.max() * 1.25


def _notes(values, log):
    notes = []
    missing = np.count_nonzero(~np.isfinite(values))
    if missing:
        notes.append(f"{missing:,} unscored LS")
    if log and (values == 0).any():
        notes.append(f"{np.count_nonzero(values == 0):,} LS at zero")
    if not log and (values > 5).any():
        notes.append(f"{np.count_nonzero(values > 5):,} LS above 5")
    return "; ".join(notes)


def _axis(ax, frame, values, *, log=False, regions=False):
    runs, positions = [], []
    for (_, run), group in frame.groupby(["dataset", "run_number"], sort=False):
        first, last = group.index[0], group.index[-1]
        color = {"good": GREEN, "bad": RED}.get(group.label.iloc[0], GREY)
        ax.axvspan(first - .5, last + .5, color=color, zorder=0)
        runs.append(run)
        positions.append((first + last) / 2)
    visible = np.isfinite(values) & ((values > 0) if log else True)
    ax.scatter(np.flatnonzero(visible), values[visible], s=9, color=BLUE,
               alpha=.5, linewidths=0, rasterized=True)
    ax.axhline(1, color="black", linestyle="--")
    if runs:
        ticks = {0, len(runs) - 1}
        ticks.update(np.argmin(np.abs(np.asarray(positions) - x))
                     for x in np.linspace(0, len(frame) - 1, 8))
        ax.set_xticks([positions[i] for i in sorted(ticks)], [str(runs[i]) for i in sorted(ticks)])
    if regions:
        for name, members in REGIONS.items():
            selected = frame.index[frame.run_number.isin(members)]
            if len(selected):
                first, last = selected[0], selected[-1]
                ax.axvspan(first - .5, last + .5, fill=False, edgecolor="#b2182b", linestyle="--")
                ax.text((first + last) / 2, .03, name, transform=ax.get_xaxis_transform(),
                        ha="center", color="#b2182b",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    notes = _notes(values, log)
    if notes:
        ax.text(.01, .78, notes, transform=ax.transAxes, va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    ax.set(xlim=(-.5, max(.5, len(frame) - .5)), ylim=_limits(values) if log else (0, 5),
           xlabel="Run number", ylabel="Anomaly score / threshold")
    if log:
        ax.set_yscale("log")


def _handles(unknown=False):
    handles = [Patch(facecolor=GREEN, edgecolor="#5a9e68", label="Good"),
            Patch(facecolor=RED, edgecolor="#be6868", label="Bad"),
            Line2D([], [], color="black", linestyle="--", label="Threshold"),
            Line2D([], [], color=BLUE, marker="o", linestyle="none", label="LS")]
    if unknown:
        handles.append(Patch(facecolor=GREY, edgecolor="grey", label="Unknown"))
    return handles


def score_view(frame, threshold, path, *, label="Test", log=False, regions=False):
    """Keep unscored rows on the LS axis; only DCS-off rows are excluded."""
    hep.style.use("CMS")
    frame = _ordered(frame)
    fig, ax = plt.subplots(figsize=(11, 6.2), layout="constrained")
    _axis(ax, frame, _values(frame, threshold), log=log, regions=regions)
    ax.legend(handles=_handles(frame.label.eq("unknown").any()), loc="upper right", ncol=4)
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel=label)
    save(fig, path)


def _distribution(frame, threshold, path, log):
    values = _values(frame, threshold)
    limits = _limits(values) if log else (0, 5)
    bins = np.geomspace(*limits, 61) if log else np.linspace(*limits, 61)
    fig, ax = plt.subplots(figsize=(9, 6.5), layout="constrained")
    for name, mask, color in (("Good", frame.label.eq("good"), "#27834b"),
                               ("Bad", frame.label.eq("bad"), "#c43c3c"),
                               ("Unknown", frame.label.eq("unknown"), "grey")):
        selected = values[mask]
        scored = np.isfinite(selected)
        visible = scored & ((selected > 0) if log else True)
        if scored.any():
            label = f"{name} ({scored.sum():,} scored, {(~scored).sum():,} unscored)"
            ax.hist(selected[visible], bins=bins, weights=np.full(visible.sum(), 1 / scored.sum()),
                    histtype="step", linewidth=2, color=color, label=label)
    ax.axvline(1, color="black", linestyle="--", label="Threshold")
    ax.set(xlabel="Anomaly score / threshold", ylabel="Fraction of scored LS / bin", xlim=limits)
    if log:
        ax.set_xscale("log")
    ax.legend()
    notes = _notes(values, log)
    if notes:
        fig.supxlabel(notes)
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel="Test")
    save(fig, path)


def plot_stage(scores, history, threshold, folder):
    """Write test views and, when supplied, one training/validation loss curve."""
    hep.style.use("CMS")
    folder = Path(folder)
    test = _ordered(scores.loc[scores.split.eq("test")])
    for log in (False, True):
        suffix = "_log" if log else ""
        score_view(test, threshold, folder / f"test{suffix}.png", log=log, regions=True)
        _distribution(test, threshold, folder / f"distributions{suffix}.png", log)
    if history is not None:
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        for key, label in (("train", "Train"), ("validation", "Validation")):
            values = history[key]
            ax.plot(np.arange(1, len(values) + 1), values, label=label)
        if history.get("best_epoch"):
            ax.axvline(history["best_epoch"], color="black", linestyle="--", label="Best epoch")
        ax.set(xlabel="Epoch", ylabel=history.get("loss", "Loss"))
        ax.legend()
        hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel="2025")
        save(fig, folder / "loss.png")


def read_scores(folder):
    """Use current RR display labels without changing stored scores or splits."""
    scores = pd.read_parquet(folder / "scores.parquet")
    try:
        labels = load_run_labels()
    except FileNotFoundError:
        scores.loc[scores.label.str.startswith("bad_", na=False), "label"] = "bad"
    else:
        scores["label"] = scores.run_number.map(labels).fillna("unknown")
    return scores


def history_for(checkpoint):
    from scripts.models.core import load_checkpoint

    spec, saved, _ = load_checkpoint(checkpoint)
    if spec.KIND == "window":
        return None
    if spec.KIND == "feature":
        return dict(train=saved["train_losses"], validation=saved["validation_losses"],
                    loss="Feature MSE")
    return saved["history"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Prediction directory")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint file or directory for the loss curve")
    parser.add_argument("--output", type=Path, help="Plot directory; default: output/plots/<input name>")
    args = parser.parse_args()
    metrics = json.loads((args.input / "metrics.json").read_text())
    checkpoint = args.checkpoint
    if checkpoint is None:
        candidate = (args.input / metrics["checkpoint"] if metrics.get("checkpoint")
                     else OUT / "model" / (metrics.get("experiment") or args.input.name))
        if candidate.exists():
            checkpoint = candidate
    history = history_for(checkpoint) if checkpoint is not None else None
    output = args.output or OUT / "plots" / args.input.name
    plot_stage(read_scores(args.input), history, metrics["threshold"], output)
    print("Saved plots to", output)


if __name__ == "__main__":
    main()
