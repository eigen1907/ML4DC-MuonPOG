"""Read the cached eight Muon MEs, preserving the historical run split."""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (CACHED_MES, DCS_PATH, IDS, MANIFEST, MODEL_MES, PROMPT_RECO_PATTERN,
                    RAW, REGIONS, RUN_LABELS, SEED, SPECIAL_RUNS, raw_folder)


def _integers(values, dtype, description):
    values = np.asarray(values)
    if (not np.isfinite(values).all() or (values < 0).any()
            or (values != np.floor(values)).any()
            or int(values.max(initial=0)) > np.iinfo(dtype).max):
        raise ValueError(f"Expected finite nonnegative integer {description}")
    return values.astype(dtype)


def _moments(counts, low, high, eta=False):
    """Historical 0–1 moments; eta is folded and an empty ME gives two zeros."""
    low, high = float(low), float(high)
    edges = np.linspace(low, high, counts.shape[1] + 1, dtype=np.float32)
    centers = (edges[:-1] + edges[1:]) / 2
    positions = np.abs(centers) / max(abs(low), abs(high)) if eta else (centers - low) / (high - low)
    counts = counts.astype(np.float64)
    totals = counts.sum(axis=1)
    weights = np.divide(counts, totals[:, None], out=np.zeros_like(counts), where=totals[:, None] > 0)
    mean = weights @ positions
    std = np.sqrt((weights * (positions - mean[:, None]) ** 2).sum(axis=1))
    return np.column_stack((mean, std))


def load_dcs():
    intervals = json.loads(DCS_PATH.read_text())
    if not isinstance(intervals, dict) or not intervals:
        raise ValueError("Expected a nonempty CMS run/LS DCS JSON")
    for run, ranges in intervals.items():
        if not str(run).isdigit() or int(run) <= 0 or not isinstance(ranges, list):
            raise ValueError("Invalid DCS run")
        for pair in ranges:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(x) is not int for x in pair) or not 0 < pair[0] <= pair[1]):
                raise ValueError("Invalid inclusive DCS LS range")
    return intervals


def select(frame, intervals, phi_entries):
    """Select DCS-ready LS; retain phi ROOT fills as metadata, including zero."""
    entries = np.asarray(phi_entries)
    if entries.shape != (len(frame),) or not np.isfinite(entries).all() or (entries < 0).any():
        raise ValueError("Expected one nonnegative phi entry count per LS")
    runs, ls = frame.run_number.to_numpy(), frame.ls_number.to_numpy()
    passed, present = np.zeros(len(frame), dtype=bool), np.zeros(len(frame), dtype=bool)
    for run in np.unique(runs):
        rows = runs == run
        present[rows] = str(int(run)) in intervals
        for first, last in intervals.get(str(int(run)), []):
            passed |= rows & (ls >= first) & (ls <= last)
    result = pd.DataFrame({"phi_entries": entries, "dcs_run_present": present,
                           "dcs_pass": passed}, index=frame.index)
    result["exclusion"] = np.select(
        [~present, ~passed], ["DCS run absent", "DCS LS excluded"], default="Included")
    return result


def load_dataset():
    """Read each selected parquet once; require all ten cached MEs per run.

    Split the full run catalogue before excluding incomplete caches. Native
    signed bins, ROOT entries and in-range bin totals remain distinct. Only the
    historical moment representation folds eta. No DIALS calls are made.
    """
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest.dataset.str.contains(PROMPT_RECO_PATTERN)].copy()
    labels = pd.read_csv(RUN_LABELS)
    runs, listed = set(manifest.run_number), set(labels.run_number)
    bad = set(labels.loc[labels.label.eq("bad_well_known"), "run_number"])
    ambiguous = set(labels.loc[labels.label.eq("bad_ambiguous"), "run_number"])
    if (manifest.empty or len(manifest) != len(runs) or len(labels) != len(listed) or runs != listed
            or not labels.label.isin({"good", "bad_well_known", "bad_ambiguous"}).all() or not bad or not ambiguous):
        raise ValueError("PromptReco runs and data/run_labels.csv do not agree")
    manifest["era"] = manifest.dataset.str.extract(r"Run(2025[A-Z])")[0]
    good = manifest[~manifest.run_number.isin(bad | ambiguous)]
    train, holdout = train_test_split(good, train_size=.70, random_state=SEED, stratify=good.era)
    validation, test = train_test_split(holdout, train_size=.5, random_state=SEED, stratify=holdout.era)
    split = {"train": set(train.run_number), "validation": set(validation.run_number),
             "test": set(test.run_number) | bad | ambiguous}
    ready = [all((raw_folder(row.dataset, row.run_number, root=RAW) / f"{stem}.parquet").exists()
                 for stem in CACHED_MES) for row in manifest.itertuples()]
    expected_runs = len(manifest)
    manifest = manifest.loc[ready]
    if manifest.empty:
        raise ValueError("No complete PromptReco runs in the raw cache")
    print(f"Using {len(manifest)}/{expected_runs} complete PromptReco runs from the current cache", flush=True)
    total_rows = int(manifest.ls_count.sum())
    n_mes = len(MODEL_MES)
    eta_me = next(me for me in MODEL_MES if me.endswith("__GlbMuon_Glb_eta"))
    phi_me = next(me for me in MODEL_MES if me.endswith("__GlbMuon_Glb_phi"))
    histograms, axes = [None] * n_mes, [None] * n_mes
    totals = np.empty((total_rows, n_mes), dtype=np.int64)
    root_entries = np.empty((total_rows, n_mes), dtype=np.int64)
    features = np.empty((total_rows, 2 * n_mes), dtype=np.float64)
    frames, start = [], 0
    for index, row in enumerate(manifest.itertuples(), 1):
        folder = raw_folder(row.dataset, row.run_number, root=RAW)
        expected_ls, entries = None, {}
        stop = start + int(row.ls_count)
        for target, stem in enumerate(MODEL_MES):
            path = folder / f"{stem}.parquet"
            cached = pd.read_parquet(path, columns=["ls_number", "entries", "data", "x_min", "x_max", "x_bin"])
            cached = cached.sort_values("ls_number")
            ls = _integers(cached.ls_number.to_numpy(), np.int64, f"LS identifiers: {path}")
            if len(ls) == 0 or (ls == 0).any() or len(np.unique(ls)) != len(ls):
                raise ValueError(f"Empty, nonpositive or duplicate LS: {path}")
            if len(ls) != row.ls_count:
                raise ValueError(f"DIALS LS count differs for run {row.run_number}")
            if expected_ls is None:
                expected_ls = ls
            elif not np.array_equal(ls, expected_ls):
                raise ValueError(f"ME LS coverage differs: {path}")
            try:
                contents = np.stack(cached.data.to_numpy())
            except ValueError as error:
                raise ValueError(f"Inconsistent histogram lengths: {path}") from error
            if contents.ndim != 2 or contents.shape[1] == 0:
                raise ValueError(f"Invalid histogram dimensions: {path}")
            contents = _integers(contents, np.int32, f"histogram contents: {path}")
            fills = _integers(cached.entries.to_numpy(), np.int64, f"ROOT entries: {path}")
            axis = cached[["x_min", "x_max", "x_bin"]].to_numpy(dtype=np.float64)
            if (not np.isfinite(axis).all() or not (axis == axis[0]).all()
                    or axis[0, 0] >= axis[0, 1] or axis[0, 2] != contents.shape[1]):
                raise ValueError(f"Invalid or changing histogram axis: {path}")
            low, high, bins = axis[0]
            if axes[target] is None:
                axes[target] = {"low": float(low), "high": float(high), "stem": stem,
                                "edges": np.linspace(low, high, int(bins) + 1)}
                histograms[target] = np.empty((total_rows, int(bins)), dtype=np.int32)
            elif (low != axes[target]["low"] or high != axes[target]["high"]
                  or bins != histograms[target].shape[1]):
                raise ValueError(f"Histogram axis differs between runs: {path}")
            histograms[target][start:stop] = contents
            totals[start:stop, target] = contents.sum(axis=1, dtype=np.int64)
            root_entries[start:stop, target] = fills
            features[start:stop, 2*target:2*target+2] = _moments(
                contents, low, high, eta=stem.endswith("__GlbMuon_Glb_eta"))
            entries[stem] = fills
        global_entries = [fills for me, fills in entries.items()
                          if me.startswith(("Muons__Isolation__global__", "Muons__MuonRecoAnalyzer__GlbMuon_Glb_"))]
        if any(not np.array_equal(global_entries[0], other) for other in global_entries[1:]):
            raise ValueError(f"Shared GlbMuon ROOT entries differ: run {row.run_number}")
        frame = pd.DataFrame({"dataset": row.dataset, "run_number": row.run_number,
                              "ls_number": expected_ls, "era": row.era, "eta_entries": entries[eta_me]})
        frame["label"] = "bad_ambiguous" if row.run_number in ambiguous else "bad_well_known" if row.run_number in bad else "good"
        frame["split"] = next(name for name, members in split.items() if row.run_number in members)
        frames.append(frame)
        start = stop
        if index % 100 == 0:
            print(f"Loaded {index}/{len(manifest)} runs", flush=True)
    frame = pd.concat(frames, ignore_index=True)[IDS + ["era", "label", "split", "eta_entries"]]
    frame = frame.join(select(frame, load_dcs(), root_entries[:, MODEL_MES.index(phi_me)]))
    return {"frame": frame, "histograms": histograms, "totals": totals,
            "root_entries": root_entries, "axes": axes,
            "features": features, "me_names": list(MODEL_MES),
            "run_split": {name: sorted(int(run) for run in members & set(manifest.run_number))
                          for name, members in split.items()}}


def validation_roles(frame):
    """Historical run-separated validation halves, then remove special controls."""
    runs = frame.loc[frame.split.eq("validation") & frame.label.eq("good"),
                     ["run_number", "era"]].drop_duplicates()
    monitoring, calibration = train_test_split(runs, test_size=.5, random_state=SEED, stratify=runs.era)
    return {"monitor_runs": sorted(int(run) for run in monitoring.run_number if run not in SPECIAL_RUNS),
            "calibration_runs": sorted(int(run) for run in calibration.run_number if run not in SPECIAL_RUNS)}


def fingerprint(dataset):
    """Identify the actual counts, historical split/labels and DCS selection."""
    digest = hashlib.sha256(DCS_PATH.read_bytes())
    digest.update(dataset["frame"][IDS + ["split", "label"]].to_csv(index=False).encode())
    for hist in dataset["histograms"]:
        digest.update(memoryview(np.ascontiguousarray(hist)))
    return digest.hexdigest()
