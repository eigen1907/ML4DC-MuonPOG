"""Read cached Muon MEs and annual Run Registry metadata for run labels."""

import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (DCS_PATH, IDS, MANIFEST, MODEL_MES, PROMPT_RECO_PATTERN,
                    RAW, REGIONS, RR_DATA, SEED, SPECIAL_RUNS, raw_folder)
from fetch_metadata import rr_verdict


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


def load_run_labels():
    """Derive Muon verdicts from annual RR JSON; absence never means Good."""
    files = sorted(RR_DATA.glob("[0-9][0-9][0-9][0-9].json"))
    if not files:
        raise FileNotFoundError("No annual RR JSON; collect RR metadata first")
    rows = []
    for path in files:
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict) or not isinstance(payload.get("runs"), list):
            raise ValueError("Expected annual RR JSON with a runs array")
        for run in payload["runs"]:
            if not isinstance(run, dict) or "run_number" not in run:
                raise ValueError("Invalid RR run record")
            components = run.get("components") or {}
            if not isinstance(components, dict):
                raise ValueError("Expected RR component verdict counts")
            counts = components.get("muon") or {}
            if not isinstance(counts, dict):
                raise ValueError("Expected RR component verdict counts")
            rows.append(dict(run_number=run["run_number"],
                             label="unknown" if run.get("deleted") else rr_verdict(counts)))
    labels = pd.DataFrame(rows, columns=["run_number", "label"])
    labels["run_number"] = _integers(pd.to_numeric(labels.run_number, errors="coerce"),
                                    np.int64, "RR run numbers")
    if ((labels.run_number <= 0).any() or labels.run_number.duplicated().any()
            or not labels.label.isin(("good", "bad", "unknown")).all()):
        raise ValueError("Expected unique run numbers and good/bad/unknown RR labels")
    return labels.set_index("run_number").label


def load_dataset():
    """Read each selected parquet once; skip runs missing a selected ME.

    Split the full run catalogue before excluding incomplete caches. Native
    signed bins, ROOT entries and in-range bin totals remain distinct. Only the
    historical moment representation folds eta. No DIALS calls are made.
    """
    manifest = pd.read_csv(MANIFEST)
    manifest = manifest[manifest.dataset.str.contains(PROMPT_RECO_PATTERN)].copy()
    if manifest.empty or manifest.run_number.duplicated().any():
        raise ValueError("Expected one cached PromptReco dataset per run")
    labels = load_run_labels()
    manifest["label"] = manifest.run_number.map(labels).fillna("unknown")
    manifest["era"] = manifest.dataset.str.extract(r"Run(2025[A-Z])")[0]
    good = manifest[manifest.label.eq("good")]
    try:
        train, holdout = train_test_split(good, train_size=.70, random_state=SEED, stratify=good.era)
        validation, test = train_test_split(holdout, train_size=.5, random_state=SEED, stratify=holdout.era)
    except ValueError as error:
        raise ValueError("Not enough RR-labelled Good runs for the run-separated split; collect more RR metadata") from error
    split = {"train": set(train.run_number), "validation": set(validation.run_number),
             "test": set(test.run_number) | set(manifest.loc[manifest.label.ne("good"), "run_number"])}
    ready = [all((raw_folder(row.dataset, row.run_number, root=RAW) / f"{stem}.parquet").exists()
                 for stem in MODEL_MES) for row in manifest.itertuples()]
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
        frame["label"] = row.label
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
    """Separate validation runs for monitoring and calibration; remove special controls."""
    runs = frame.loc[frame.split.eq("validation") & frame.label.eq("good"),
                     ["run_number", "era"]].drop_duplicates()
    monitoring, calibration = train_test_split(runs, test_size=.5, random_state=SEED, stratify=runs.era)
    return {"monitor_runs": sorted(int(run) for run in monitoring.run_number if run not in SPECIAL_RUNS),
            "calibration_runs": sorted(int(run) for run in calibration.run_number if run not in SPECIAL_RUNS)}


def fingerprint(dataset):
    """Identify actual counts, the current run split/RR labels and DCS selection."""
    digest = hashlib.sha256(DCS_PATH.read_bytes())
    digest.update(dataset["frame"][IDS + ["split", "label"]].to_csv(index=False).encode())
    for hist in dataset["histograms"]:
        digest.update(memoryview(np.ascontiguousarray(hist)))
    return digest.hexdigest()
