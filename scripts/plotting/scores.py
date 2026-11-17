"""Plot saved LS scores and optional training losses; no model fitting."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import pandas as pd

from scripts.models.config import IDS, MAIN_PREDICTIONS, OUT
from scripts.data.dials import load_run_labels
from scripts.models.preprocessing import load_muon_certification


LS_COLORS = {"Good": "#27834b", "JSON Bad": "#e69f00",
             "DCS Bad": "#8056b3", "RR Unknown": "#0072b2",
             "RR Bad": "#c43c3c"}
LINEAR_SCORE_QUANTILE = .999


def quality_labels(frame):
    """Exclusive reasons for the operational Good/Bad quality labels.

    Overlap priority: RR Bad, DCS Bad, JSON Bad, then RR status.
    Keep the saved labels/scores intact; absent RR verdicts never become Good.
    """
    if 'quality_label' in frame:
        return frame.quality_label
    rr = frame.get('rr_label', frame['label'])
    dcs = frame.dcs_pass.fillna(False).to_numpy(dtype=bool)
    certified = (frame.muon_certified.fillna(False).to_numpy(dtype=bool)
                 if 'muon_certified' in frame else load_muon_certification(frame))
    labels = np.select([rr.eq('bad'), ~dcs, ~certified, rr.eq('good')],
                       ['RR Bad', 'DCS Bad', 'JSON Bad', 'Good'],
                       default='RR Unknown')
    return pd.Series(labels, index=frame.index, name='quality_label')


def save(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)



def _ordered(scores):
    if scores.duplicated(IDS).any():
        raise ValueError("Duplicate dataset/run/LS rows")
    selected = scores if 'quality_pass' in scores else scores.loc[scores.dcs_pass.fillna(False)]
    return selected.sort_values(
        ["run_number", "ls_number", "dataset"]).reset_index(drop=True)


def _values(frame, threshold):
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("A positive finite threshold is required")
    values = frame.anomaly_score.to_numpy(dtype=float, na_value=np.nan) / threshold
    if (values[np.isfinite(values)] < 0).any():
        raise ValueError("Anomaly scores must be nonnegative")
    return values


def _limits(values, *, log=False):
    finite = values[np.isfinite(values)]
    if log:
        positive = np.r_[1., finite[finite > 0]]
        return positive.min() * .8, positive.max() * 1.25
    upper = np.quantile(finite, LINEAR_SCORE_QUANTILE) if len(finite) else 1
    return 0, max(1.1, upper)


def _years(frame):
    return sorted(frame.dataset.str.extract(r"/Run(20\d\d)", expand=False).dropna().unique())


def _period(years):
    years = sorted({str(year) for year in years})
    if len(years) > 1 and int(years[-1]) - int(years[0]) + 1 == len(years):
        return f"{years[0]}–{years[-1]}"
    return ", ".join(years)


def _axis(ax, frame, values, *, log=False):
    datasets = frame.dataset.to_numpy()
    years = frame.dataset.str.extract(r'/Run(20\d\d)', expand=False).fillna('').to_numpy()
    for first in np.flatnonzero(datasets[1:] != datasets[:-1]) + 1:
        year_change = years[first] != years[first - 1]
        ax.axvline(first - .5, color='0.5' if year_change else '0.75',
                   linewidth=1.4 if year_change else .6, alpha=.8, zorder=0)
    runs, positions = [], []
    for run, group in frame.groupby("run_number", sort=False):
        first, last = group.index[0], group.index[-1]
        runs.append(run)
        positions.append((first + last) / 2)
    visible = np.isfinite(values) & ((values > 0) if log else True)
    labels = quality_labels(frame)
    for status, color in LS_COLORS.items():
        selected = visible & labels.eq(status).to_numpy()
        if not labels.eq(status).any():
            continue
        ax.scatter(np.flatnonzero(selected), values[selected], s=9, color=color,
                   alpha=.9, linewidths=0, rasterized=True, label=status)
    ax.axhline(1, color="black", linestyle="--")
    if runs:
        # Use a few well-spaced run ticks while keeping consecutive LS positions.
        gap = len(frame) / 8
        ticks = [0]
        for index in range(1, len(runs) - 1):
            if positions[index] - positions[ticks[-1]] >= gap and positions[-1] - positions[index] >= gap:
                ticks.append(index)
        if len(runs) > 1:
            ticks.append(len(runs) - 1)
        ax.set_xticks([positions[i] for i in ticks], [str(runs[i]) for i in ticks])
    ax.tick_params(axis="x", which="both", top=False)
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set(xlim=(-.5, max(.5, len(frame) - .5)), ylim=_limits(values, log=log),
           xlabel="Run", ylabel="Score")
    if log:
        ax.set_yscale("log")


def score_view(frame, threshold, path, *, label="Test", log=False):
    """Keep all quality-policy Test rows; legacy views retain their DCS filter."""
    hep.style.use("CMS")
    frame = _ordered(frame)
    fig, ax = plt.subplots(figsize=(14, 8.5), layout="constrained")
    _axis(ax, frame, _values(frame, threshold), log=log)
    ax.legend(loc="upper right", ncol=2, markerscale=2, frameon=True)
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel=label)
    save(fig, path)


def _distribution(frame, threshold, path, log, *, label="Test"):
    values = _values(frame, threshold)
    limits = _limits(values, log=log)
    bins = np.geomspace(*limits, 61) if log else np.linspace(*limits, 61)
    fig, ax = plt.subplots(figsize=(11, 7.5), layout="constrained")
    labels = quality_labels(frame)
    for status, color in LS_COLORS.items():
        selected = values[labels.eq(status)]
        scored = np.isfinite(selected)
        visible = scored & ((selected > 0) if log else True)
        if scored.any():
            legend_label = f"{status} ({scored.sum():,})"
            ax.hist(selected[visible], bins=bins, weights=np.full(visible.sum(), 1 / scored.sum()),
                    histtype="step", linewidth=2, color=color, label=legend_label)
    ax.axvline(1, color="black", linestyle="--")
    ax.set(xlabel="Score", ylabel="Fraction", xlim=limits)
    if log:
        ax.set_xscale("log")
    ax.legend()
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel=label)
    save(fig, path)


def stat_score_views(frame, threshold, folder, *, label="Test"):
    """Scatter retained Test LS scores against ROOT Global-muon eta entries."""
    hep.style.use("CMS")
    folder = Path(folder)
    values = _values(frame, threshold)
    entries = frame.eta_entries.to_numpy(dtype=float, na_value=np.nan)
    if (entries[np.isfinite(entries)] < 0).any():
        raise ValueError("Global muon eta entries must be nonnegative")
    finite = np.isfinite(entries) & np.isfinite(values)
    x_high = max(1, float(entries[finite].max()) if finite.any() else 0)
    grouped = 'n_ls' in frame
    x_low = .9 * float(entries[finite].min()) if grouped and finite.any() else 0
    y_high = max(1.1, float(values[finite].max()) if finite.any() else 0)
    labels = quality_labels(frame).to_numpy()
    for name, category in (("all", None), ("good", "Good"), ("rr_bad", "RR Bad"),
                           ("json_bad", "JSON Bad"), ("dcs_bad", "DCS Bad")):
        fig, ax = plt.subplots(figsize=(14, 8.5), layout="constrained")
        statuses = LS_COLORS if category is None else {category: LS_COLORS[category]}
        for status, color in statuses.items():
            selected = finite & (labels == status)
            if category is None and not selected.any():
                continue
            ax.scatter(entries[selected], values[selected], s=9, color=color,
                       alpha=.9, linewidths=0, rasterized=True, label=status)
        if grouped:
            ax.set_xscale("log")
        else:
            ax.set_xscale("symlog", linthresh=1)
        ax.set_yscale("symlog", linthresh=.01)
        ax.axhline(1, color="black", linestyle="--")
        ax.set(xlabel="Entries", ylabel="Score", xlim=(x_low, x_high), ylim=(0, y_high))
        if ax.get_legend_handles_labels()[0]:
            ax.legend(loc="upper right", markerscale=2)
        hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel=label)
        save(fig, folder / f"stat_score_{name}.png")
    (folder / "stat_score.png").unlink(missing_ok=True)


def confusion_view(frame, threshold, path, *, label="Test"):
    """Flag fractions per exclusive quality source under the saved label policy."""
    values = frame.anomaly_score.to_numpy(dtype=float, na_value=np.nan)
    labels = quality_labels(frame)
    statuses = [status for status in LS_COLORS if labels.eq(status).any()]
    counts = np.array([[int((labels.eq(status) & np.isfinite(values) & (values <= threshold)).sum()),
                        int((labels.eq(status) & np.isfinite(values) & (values > threshold)).sum())]
                       for status in statuses], dtype=int).reshape(-1, 2).T
    totals = counts.sum(axis=0, keepdims=True)
    fractions = np.divide(counts, totals, out=np.zeros_like(counts, dtype=float), where=totals > 0)
    fig, ax = plt.subplots(figsize=(14, 7.5), layout="constrained")
    artist = ax.imshow(fractions if statuses else np.zeros((2, 1)),
                       cmap="Blues", vmin=0, vmax=1, aspect='auto')
    fig.colorbar(artist, ax=ax)
    for row in range(2):
        for column in range(len(statuses)):
            percent = f"{fractions[row, column]:.1%}" if totals[0, column] else "n/a"
            ax.text(column, row, f"{counts[row, column]:,}\n{percent}", ha="center", va="center",
                    color="white" if fractions[row, column] > .6 else "black")
    ax.set(xticks=np.arange(len(statuses)), xticklabels=statuses,
           yticks=[0, 1], yticklabels=["Not flagged", "Flagged"])
    ax.tick_params(which="both", length=0)
    hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel=label)
    save(fig, path)


def plot_stage(scores, history, threshold, folder):
    """Write the complete Test cohort and an optional training loss curve."""
    hep.style.use("CMS")
    scores = scores.assign(quality_label=quality_labels(scores))
    folder = Path(folder)
    test = _ordered(scores.loc[scores.split.eq("test")])
    period = _period(_years(test))
    label = f"{period} Test" if period and period != "2025" else "Test"
    cohort = (history or {}).get('cohort_label') or ('Grouped LS' if 'n_ls' in scores else None)
    if cohort:
        label = label.replace('Test', f'Test ({cohort})', 1)
    for log in (False, True):
        suffix = "_log" if log else ""
        score_view(test, threshold, folder / f"test{suffix}.png", label=label, log=log)
        _distribution(test, threshold, folder / f"distributions{suffix}.png", log, label=label)
    confusion_view(test, threshold, folder / "confusion_matrix.png", label=label)
    stat_score_views(test, threshold, folder, label=label)
    if history is not None:
        fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")
        for key, label in (("train", "Train"), ("validation", "Validation")):
            values = history[key]
            ax.plot(np.arange(1, len(values) + 1), values, label=label)
        if history.get("best_epoch"):
            ax.axvline(history["best_epoch"], color="black", linestyle="--", label="Best epoch")
        ax.set(xlabel="Epoch", ylabel="Loss")
        ax.legend()
        years = history.get("training_years")
        if years is None:
            years = _years(scores.loc[scores.split.isin(("train", "validation"))])
        period = _period(years)
        label = period or "Training"
        hep.cms.label(ax=ax, text="Work in Progress", data=True, com=None, rlabel=label)
        save(fig, folder / "loss.png")
    # Remove superseded generated views only after every replacement succeeded.
    for name in ("all", "nover1000", "nover5000", "no2026C", "nolowstats"):
        if (folder / name).is_dir():
            shutil.rmtree(folder / name)



def read_scores(folder):
    """Use current RR display labels without changing stored scores or splits."""
    scores = pd.read_parquet(folder / "scores.parquet")
    if 'quality_pass' in scores:
        return scores
    try:
        labels = load_run_labels()
    except FileNotFoundError:
        scores.loc[scores.label.str.startswith("bad_", na=False), "label"] = "bad"
    else:
        scores["label"] = scores.run_number.map(labels).fillna("unknown")
    return scores


def history_for(checkpoint):
    from scripts.models.core import load_checkpoint

    _, saved, _ = load_checkpoint(checkpoint)
    history = dict(saved["history"])
    history["training_years"] = saved.get("training_years") or (saved.get("signature") or {}).get("training_years")
    if saved.get('ls_grouping'):
        history['cohort_label'] = 'Grouped LS'
    return history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", default=[MAIN_PREDICTIONS], help="One or more prediction directories from the same checkpoint")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint file or directory for the loss curve")
    parser.add_argument("--output", type=Path, help="Plot directory; default: mirror the input path under output/plots")
    args = parser.parse_args()
    if len(args.input) > 1 and args.output is None:
        parser.error("Use --output when combining prediction directories")
    records = [json.loads((folder / "metrics.json").read_text()) for folder in args.input]
    metrics = records[0]
    candidates = [(folder / record["checkpoint"] if record.get("checkpoint")
                   else OUT / "model" / (record.get("experiment") or folder.name))
                  for folder, record in zip(args.input, records)]
    if any(record.get(key) != metrics.get(key) for record in records[1:]
           for key in ("experiment", "threshold", "quantile", "signature", "split_policy")):
        raise ValueError("Combined predictions must share the experiment, calibration and training signature")
    if any(candidate.resolve() != candidates[0].resolve() for candidate in candidates[1:]):
        raise ValueError("Combined predictions must use the same checkpoint")
    scores = pd.concat([read_scores(folder) for folder in args.input], ignore_index=True)
    if scores.duplicated(IDS).any():
        raise ValueError("Duplicate dataset/run/LS rows across prediction directories")
    checkpoint = args.checkpoint
    if checkpoint is None and candidates[0].exists():
        checkpoint = candidates[0]
    history = (history_for(checkpoint) if checkpoint is not None
               and metrics.get('experiment') != '00_ref_score' else None)
    if args.output is not None:
        output = args.output
    else:
        try:
            relative = args.input[0].resolve().relative_to((OUT / "predict").resolve())
        except ValueError:
            relative = Path(args.input[0].name)
        output = OUT / 'plots' / relative
    plot_stage(scores, history, metrics["threshold"], output)
    print("Saved plots to", output)


if __name__ == "__main__":
    main()
