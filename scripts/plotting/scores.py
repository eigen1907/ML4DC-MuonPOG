"""Plot saved LS scores and optional training losses; no model fitting."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

from scripts.models.config import IDS, OUT
from scripts.data.dials import load_run_labels


LS_COLORS = {"good": "#1769c2", "bad": "#c43c3c", "unknown": "#777777"}


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


def _axis(ax, frame, values, *, log=False):
    runs, positions = [], []
    for run, group in frame.groupby("run_number", sort=False):
        first, last = group.index[0], group.index[-1]
        if first:
            ax.axvline(first - .5, color="0.85", linewidth=.5, zorder=0)
        runs.append(run)
        positions.append((first + last) / 2)
    visible = np.isfinite(values) & ((values > 0) if log else True)
    for status, color in LS_COLORS.items():
        selected = visible & frame.label.eq(status).to_numpy()
        ax.scatter(np.flatnonzero(selected), values[selected], s=9, color=color,
                   alpha=.65, linewidths=0, rasterized=True, label=f"{status.title()} LS")
    ax.axhline(1, color="black", linestyle="--", label="Threshold")
    if runs:
        # Short runs remain visible, but nearby run labels must not overlap.
        gap = len(frame) / 20
        ticks = [0]
        for index in range(1, len(runs) - 1):
            if positions[index] - positions[ticks[-1]] >= gap and positions[-1] - positions[index] >= gap:
                ticks.append(index)
        if len(runs) > 1:
            ticks.append(len(runs) - 1)
        top = ax.secondary_xaxis("top")
        top.set_xticks([positions[i] for i in ticks], [str(runs[i]) for i in ticks], rotation=90)
        top.set_xlabel("Run number")
        ax.tick_params(axis="x", top=False)
    notes = _notes(values, log)
    if notes:
        ax.text(.01, .78, notes, transform=ax.transAxes, va="top",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9})
    ax.set(xlim=(-.5, max(.5, len(frame) - .5)), ylim=_limits(values) if log else (0, 5),
           xlabel="Cumulative LS", ylabel="Anomaly score / threshold")
    if log:
        ax.set_yscale("log")


def score_view(frame, threshold, path, *, label="Test", log=False):
    """Keep unscored rows on the LS axis; only DCS-off rows are excluded."""
    hep.style.use("CMS")
    frame = _ordered(frame)
    fig, (header, ax) = plt.subplots(2, 1, figsize=(11, 7.5), layout="constrained",
                                     gridspec_kw={"height_ratios": [.04, 1]})
    header.set_axis_off()
    _axis(ax, frame, _values(frame, threshold), log=log)
    ax.legend(loc="upper right", ncol=4, markerscale=2)
    hep.cms.label(ax=header, text="Work in Progress", data=True, com=None, rlabel=label)
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
        score_view(test, threshold, folder / f"test{suffix}.png", log=log)
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
