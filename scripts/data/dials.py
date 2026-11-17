"""Read cached Muon MEs and annual Run Registry metadata for run labels."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from scripts.models.config import (DCS_DATA, DIALS, IDS, MODEL_MES, PROMPT_RECO_PATTERN,
                    REGIONS, RR_DATA, SEED, SPECIAL_RUNS, SPLIT_POLICY, cached_runs, dials_folder)
from scripts.models.preprocessing import _moments


def _integers(values, dtype, description):
    values = np.asarray(values)
    if (not np.isfinite(values).all() or (values < 0).any()
            or (values != np.floor(values)).any()
            or int(values.max(initial=0)) > np.iinfo(dtype).max):
        raise ValueError(f"Expected finite nonnegative integer {description}")
    return values.astype(dtype)


def _years(years):
    years = sorted(set(years))
    if not years or any(type(year) is not int or not 2000 <= year <= 2099 for year in years):
        raise ValueError("Expected one or more 20xx years")
    return years


def load_dcs(years):
    """Merge required annual snapshots; a missing year must not exclude every LS."""
    intervals = {}
    for year in _years(years):
        path = DCS_DATA / f"{year}.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing DCS metadata: {path}; collect this year first")
        annual = json.loads(path.read_text())
        if not isinstance(annual, dict) or not annual:
            raise ValueError(f"Expected a nonempty CMS run/LS DCS JSON: {path}")
        for run, ranges in annual.items():
            if not str(run).isdigit() or int(run) <= 0 or not isinstance(ranges, list):
                raise ValueError(f"Invalid DCS run: {path}")
            if run in intervals:
                raise ValueError(f"Duplicate DCS run across annual snapshots: {run}")
            for pair in ranges:
                if (not isinstance(pair, list) or len(pair) != 2
                        or any(type(x) is not int for x in pair) or not 0 < pair[0] <= pair[1]):
                    raise ValueError(f"Invalid inclusive DCS LS range: {path}")
        intervals.update(annual)
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


def rr_verdict(counts):
    """Bad takes precedence; incomplete or absent verdicts stay unknown."""
    numbers = {key: value for key, value in counts.items() if isinstance(value, (int, float))}
    if any(value < 0 for value in numbers.values()):
        return "unknown"
    if numbers.get("BAD", 0) > 0:
        return "bad"
    if numbers.get("GOOD", 0) > 0 and all(value == 0 for key, value in numbers.items() if key != "GOOD"):
        return "good"
    return "unknown"



def load_run_labels(years=None):
    """Derive Muon verdicts from annual RR JSON; absence never means Good."""
    files = (sorted(RR_DATA.glob("[0-9][0-9][0-9][0-9].json")) if years is None
             else [RR_DATA / f"{year}.json" for year in _years(years)])
    if not files:
        raise FileNotFoundError("No annual RR JSON; collect RR metadata first")
    rows = []
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing RR metadata: {path}; collect this year first")
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
                             label="unknown" if run.get("deleted") or run.get("rr_dataset") == "online"
                             else rr_verdict(counts)))
    labels = pd.DataFrame(rows, columns=["run_number", "label"])
    labels["run_number"] = _integers(pd.to_numeric(labels.run_number, errors="coerce"),
                                    np.int64, "RR run numbers")
    if ((labels.run_number <= 0).any() or labels.run_number.duplicated().any()
            or not labels.label.isin(("good", "bad", "unknown")).all()):
        raise ValueError("Expected unique run numbers and good/bad/unknown RR labels")
    return labels.set_index("run_number").label


def assign_quality_splits(frame, run_split=None, *, policy=SPLIT_POLICY):
    """Reserve Good run roles, then put every LS failing quality in Test.

    Only RR-Good, DCS-pass, Muon-certified LS with positive eta ROOT entries
    can enter Train/Validation. Zero-entry LS are operationally Bad, then
    excluded from every ML split by load_dataset. Positive low counts remain
    eligible. Run roles are reserved from RR/DCS/JSON quality before this veto.
    Fresh roles use all selected years. Supplied checkpoint roles are replayed,
    with unseen runs held out. Failed LS from a reserved fitting run are tagged
    explicitly; they are not independent held-out-run observations.
    """
    from scripts.models.preprocessing import load_muon_certification

    if policy != SPLIT_POLICY:
        raise ValueError('Unknown quality split policy')
    frame = frame.copy()
    if 'rr_label' not in frame:
        frame['rr_label'] = frame['label']
    if 'muon_certified' not in frame:
        frame['muon_certified'] = load_muon_certification(frame)
    frame['muon_certified'] = frame.muon_certified.fillna(False).astype(bool)
    quality = (frame.rr_label.eq('good') & frame.dcs_pass.fillna(False)
               & frame.muon_certified)
    positive_eta = frame.eta_entries.gt(0)
    frame['quality_pass'] = quality & positive_eta
    bad = (frame.rr_label.eq('bad') | ~frame.dcs_pass.fillna(False)
           | ~frame.muon_certified | ~positive_eta)
    frame['label'] = np.select([bad, frame.quality_pass],
                                ['bad', 'good'], default='unknown')
    observed = set(frame.run_number)
    if run_split is None:
        qualified = frame.loc[quality,
                              ['run_number', 'era']].drop_duplicates('run_number')
        try:
            train, holdout = train_test_split(qualified, train_size=.70, random_state=SEED,
                                             stratify=qualified.era)
            validation, _ = train_test_split(holdout, train_size=.5, random_state=SEED,
                                             stratify=holdout.era)
        except ValueError as error:
            raise ValueError('Not enough quality-qualified Good runs for era-stratified 70/15/15 roles') from error
        roles = {'train': set(train.run_number), 'validation': set(validation.run_number)}
        roles['test'] = observed-roles['train']-roles['validation']
    else:
        roles = {name: set(run_split[name]) for name in ('train', 'validation', 'test')}
        roles['test'] |= observed-roles['train']-roles['validation']
    if any(roles[a] & roles[b] for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test'))):
        raise ValueError('Reserved Train/Validation/Test run roles must be disjoint')
    frame['split'] = 'test'
    for name in ('train', 'validation'):
        frame.loc[frame.quality_pass & frame.run_number.isin(roles[name]), 'split'] = name
    seen_run = frame.run_number.isin(roles['train'] | roles['validation'])
    frame['test_group'] = np.where(frame.split.eq('test'),
        np.where(seen_run, 'seen_run_quality_fail', 'held_out_run'), '')
    return frame, {name: sorted(int(run) for run in members) for name, members in roles.items()}


def load_dataset(root=DIALS, *, split=True, years=None, prompt_reco=False, quality_split=False):
    """Read each selected parquet once; skip runs missing a selected ME.

    Legacy training splits discovered Muon0 PromptReco runs before excluding incomplete caches.
    Each physical run must have one processing to avoid duplicated training LS.
    Inference accepts any cached dataset unless prompt_reco is requested, with
    every row initially assigned to Test. Native
    signed bins, ROOT entries and in-range bin totals remain distinct. Moments
    retain the signed native axis. quality_split reserves roles
    only after reading LS quality and complete caches, using the selected years.
    LS with zero Global Muon eta ROOT entries are excluded from all returned
    arrays, after reserving physical-run roles. No positive-count threshold
    is applied. No DIALS calls are made and source cache files are untouched.
    """
    root = DIALS if root is None else Path(root)
    runs = pd.DataFrame(cached_runs(root), columns=["dataset", "run_number"])
    if split or prompt_reco or quality_split:
        runs = runs.loc[runs.dataset.str.contains(PROMPT_RECO_PATTERN, na=False)].copy()
    run_years = runs.dataset.str.extract(r"/Run(20\d\d)")[0]
    if years is not None:
        runs = runs.loc[run_years.isin([str(year) for year in _years(years)])].copy()
        run_years = run_years.loc[runs.index]
    if runs.empty or runs.duplicated(["dataset", "run_number"]).any():
        raise ValueError("Expected nonempty unique cached dataset/run pairs")
    if run_years.isna().any():
        raise ValueError("Cannot determine DCS year from a cached dataset name")
    selected_years = sorted(run_years.astype(int).unique().tolist())
    if (split or prompt_reco or quality_split) and runs.run_number.duplicated(keep=False).any():
        repeated = runs.loc[runs.run_number.duplicated(keep=False), ["run_number", "dataset"]]
        raise ValueError("Multiple PromptReco processings for the same run; select one cache processing "
                         "per run:\n" + repeated.to_string(index=False))
    if split or quality_split:
        labels = load_run_labels(selected_years)
    else:
        available_years = [year for year in selected_years if (RR_DATA / f"{year}.json").is_file()]
        missing_years = sorted(set(selected_years) - set(available_years))
        if missing_years:
            print(f"RR metadata unavailable for {missing_years}; their labels remain unknown", flush=True)
        labels = load_run_labels(available_years) if available_years else pd.Series(dtype="object")
    runs["label"] = runs.run_number.map(labels).fillna("unknown")
    runs["era"] = runs.dataset.str.extract(r"Run(20\d\d[A-Z])")[0].fillna("unknown")
    if split and not quality_split:
        good = runs[runs.label.eq("good")]
        try:
            train, holdout = train_test_split(good, train_size=.70, random_state=SEED, stratify=good.era)
            validation, test = train_test_split(holdout, train_size=.5, random_state=SEED, stratify=holdout.era)
        except ValueError as error:
            raise ValueError("Not enough RR-labelled Good runs for the run-separated split; collect more RR metadata") from error
        run_split = {"train": set(train.run_number), "validation": set(validation.run_number),
                     "test": set(test.run_number) | set(runs.loc[runs.label.ne("good"), "run_number"])}
    else:
        run_split = {"train": set(), "validation": set(), "test": set(runs.run_number)}
    ready = [all((dials_folder(row.dataset, row.run_number, root=root) / f"{stem}.parquet").exists()
                 for stem in MODEL_MES) for row in runs.itertuples()]
    expected_runs = len(runs)
    runs = runs.loc[ready].copy()
    if runs.empty:
        raise ValueError("No complete runs in the DIALS cache")
    selected_years = sorted(runs.dataset.str.extract(r"/Run(20\d\d)")[0].astype(int).unique().tolist())
    intervals = load_dcs(selected_years)
    runs["ls_count"] = [pq.read_metadata(dials_folder(row.dataset, row.run_number, root=root)
                                         / f"{MODEL_MES[0]}.parquet").num_rows
                        for row in runs.itertuples()]
    print(f"Using {len(runs)}/{expected_runs} complete runs from the current cache", flush=True)
    total_rows = int(runs.ls_count.sum())
    n_mes = len(MODEL_MES)
    eta_me = next(me for me in MODEL_MES if me.endswith("__GlbMuon_Glb_eta"))
    phi_me = next(me for me in MODEL_MES if me.endswith("__GlbMuon_Glb_phi"))
    histograms, axes = [None] * n_mes, [None] * n_mes
    totals = np.empty((total_rows, n_mes), dtype=np.int64)
    root_entries = np.empty((total_rows, n_mes), dtype=np.int64)
    features = np.empty((total_rows, 2 * n_mes), dtype=np.float64)
    frames, start = [], 0
    for index, row in enumerate(runs.itertuples(), 1):
        folder = dials_folder(row.dataset, row.run_number, root=root)
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
                raise ValueError(f"Cached ME LS count differs for run {row.run_number}")
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
            features[start:stop, 2*target:2*target+2] = _moments(contents, low, high)
            entries[stem] = fills
        global_entries = [fills for me, fills in entries.items()
                          if me.startswith(("Muons__Isolation__global__", "Muons__MuonRecoAnalyzer__GlbMuon_Glb_"))]
        if any(not np.array_equal(global_entries[0], other) for other in global_entries[1:]):
            raise ValueError(f"Shared GlbMuon ROOT entries differ: run {row.run_number}")
        frame = pd.DataFrame({"dataset": row.dataset, "run_number": row.run_number,
                              "ls_number": expected_ls, "era": row.era, "eta_entries": entries[eta_me]})
        frame["label"] = row.label
        frame["split"] = next(name for name, members in run_split.items() if row.run_number in members)
        frames.append(frame)
        start = stop
        if index % 100 == 0:
            print(f"Loaded {index}/{len(runs)} runs", flush=True)
    frame = pd.concat(frames, ignore_index=True)[IDS + ["era", "label", "split", "eta_entries"]]
    frame = frame.join(select(frame, intervals, root_entries[:, MODEL_MES.index(phi_me)]))
    dataset = {"frame": frame, "histograms": histograms, "totals": totals,
            "root_entries": root_entries, "axes": axes,
            "features": features, "me_names": list(MODEL_MES),
            "run_split": {name: sorted(int(run) for run in members & set(runs.run_number))
                          for name, members in run_split.items()}}
    if quality_split:
        dataset['frame'], dataset['run_split'] = assign_quality_splits(frame)
        dataset['split_policy'] = SPLIT_POLICY
    positive_eta = dataset['frame'].eta_entries.gt(0).to_numpy()
    excluded = int((~positive_eta).sum())
    dataset['excluded_zero_eta_ls'] = excluded
    if excluded:
        dataset['frame'] = dataset['frame'].loc[positive_eta].reset_index(drop=True)
        dataset['features'] = features[positive_eta]
        dataset['histograms'] = [histogram[positive_eta] for histogram in histograms]
        dataset['totals'] = totals[positive_eta]
        dataset['root_entries'] = root_entries[positive_eta]
        print(f'Excluded {excluded:,} LS with zero Global Muon eta ROOT entries', flush=True)
    return dataset


def validation_roles(frame, *, quality_split=False):
    """Separate validation runs; only the legacy policy excludes special controls."""
    selected = frame.split.eq('validation')
    if not quality_split:
        selected &= frame.label.eq('good')
    runs = frame.loc[selected,
                     ["run_number", "era"]].drop_duplicates()
    monitoring, calibration = train_test_split(runs, test_size=.5, random_state=SEED, stratify=runs.era)
    return {"monitor_runs": sorted(int(run) for run in monitoring.run_number if quality_split or run not in SPECIAL_RUNS),
            "calibration_runs": sorted(int(run) for run in calibration.run_number if quality_split or run not in SPECIAL_RUNS)}


def fingerprint(dataset):
    """Identify actual counts, labels, splits and applicable quality metadata."""
    years = sorted(dataset["frame"].dataset.str.extract(r"/Run(20\d\d)")[0].astype(int).unique())
    if len(years) == 1:
        digest = hashlib.sha256((DCS_DATA / f"{years[0]}.json").read_bytes())
    else:
        digest = hashlib.sha256()
        for year in years:
            digest.update(f"{year}:".encode())
            digest.update((DCS_DATA / f"{year}.json").read_bytes())
            digest.update(b"\0")
    columns = IDS + ['split', 'label'] + [name for name in
        ('rr_label', 'muon_certified', 'quality_pass', 'test_group') if name in dataset['frame']]
    if dataset.get('ls_grouping'):
        columns += ['ls_end', 'n_ls', 'eta_entries']
    if 'recorded_lumi_pb' in dataset['frame']:
        columns.append('recorded_lumi_pb')
    digest.update(dataset['frame'][columns].to_csv(index=False).encode())
    if dataset.get('ls_grouping'):
        digest.update(json.dumps(dataset['ls_grouping'], sort_keys=True).encode())
    if dataset.get('split_policy'):
        digest.update(dataset['split_policy'].encode())
    for hist in dataset["histograms"]:
        digest.update(memoryview(np.ascontiguousarray(hist)))
    return digest.hexdigest()
