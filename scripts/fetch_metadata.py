"""Collect DCS, OMS or Run Registry metadata as raw JSON and small CSV summaries."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from urllib.parse import parse_qsl, urlsplit

import pandas as pd
import requests


DATA = Path("data")
MANIFEST = Path("output/dataset/runs.csv")
OMS = "https://cmsoms.cern.ch/agg/api/v1/"
RR = "https://cmsrunregistry.web.cern.ch/api"
OMS_FIELDS = (
    "run_number", "lumisection_number", "fill_number", "start_time", "end_time",
    "delivered_lumi_per_lumisection", "recorded_lumi_per_lumisection", "pileup",
    "beams_stable", "cms_active", "dt0_ready", "dtm_ready", "dtp_ready",
    "rpc_ready", "cscm_ready", "cscp_ready",
)
RR_FIELDS = ("run_number", "rr_dataset", "start", "end", "status", "label", "comment", "cause")


def setup_tls():
    if sys.platform == "darwin":
        import truststore
        truststore.inject_into_ssl()


def save_response(response, path):
    """Save successful JSON response bodies byte-for-byte, without auth headers."""
    response.raise_for_status()
    if "json" not in response.headers.get("content-type", "").lower():
        raise ValueError("Expected JSON; check CERN authentication")
    payload = json.loads(response.content)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return payload, dict(file=path.name, url=response.url,
                         retrieved_at_utc=datetime.now(timezone.utc).isoformat(),
                         sha256=hashlib.sha256(response.content).hexdigest())


def write_source(folder, records, **extra):
    (folder/"source.json").write_text(json.dumps(
        dict(complete=True, requests=records, **extra), indent=2) + "\n")


def read_raw(folder):
    source = json.loads((folder/"source.json").read_text())
    if not source.get("complete", True):
        raise ValueError("Incomplete raw metadata download")
    saved = {}
    for record in source["requests"]:
        name = record["file"]
        if Path(name).name != name or name in saved:
            raise ValueError("Invalid raw filename")
        body = (folder/name).read_bytes()
        if hashlib.sha256(body).hexdigest() != record["sha256"]:
            raise ValueError(f"Raw metadata changed: {name}")
        saved[name] = json.loads(body)
    return source, saved


def write_csv(frame, source, name):
    folder = DATA/source/"summary"
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder/f"{name}.csv", index=False)
    return frame


def dcs_url(year):
    base = "https://cms-service-dqmdc.web.cern.ch/CAF/certification/"
    if year == 2023:
        return base + "Collisions23/DCSOnly_JSONS/Collisions23_13p6TeV_eraBCD_366403_370790_DCSOnly_TkPx.json"
    if year not in (2024, 2025, 2026):
        raise ValueError("Supported DCS snapshots: 2023–2026")
    short = str(year)[2:]
    return base + f"Collisions{short}/DCSOnly_JSONS/dailyDCSOnlyJSON/Collisions{short}_13p6TeV_Latest.json"


def dcs_collect(years, *, offline=False, session=None):
    """Public tracker-ready intervals, not Muon certification labels."""
    session = session or requests.Session()
    for year in years:
        raw = DATA/"dcs/raw"/f"{year}.json"
        if offline:
            body = raw.read_bytes()
            record = json.loads(raw.with_suffix(".source.json").read_text())
            if hashlib.sha256(body).hexdigest() != record["sha256"]:
                raise ValueError("Raw DCS metadata changed")
            payload = json.loads(body)
        else:
            payload, record = save_response(session.get(dcs_url(year), timeout=60), raw)
            raw.with_suffix(".source.json").write_text(json.dumps(record, indent=2) + "\n")
        if not isinstance(payload, dict) or not payload:
            raise ValueError("Expected nonempty DCS run/LS intervals")
        rows = []
        for run, ranges in payload.items():
            if not str(run).isdigit() or int(run) <= 0 or not isinstance(ranges, list):
                raise ValueError("Invalid DCS run")
            for pair in ranges:
                if (not isinstance(pair, list) or len(pair) != 2
                        or any(type(x) is not int for x in pair) or not 0 < pair[0] <= pair[1]):
                    raise ValueError("Invalid inclusive DCS LS range")
                rows.append((int(run), *pair))
        write_csv(pd.DataFrame(rows, columns=["run_number", "first_ls", "last_ls"]), "dcs", year)


def oms_pages(session, endpoint, query, folder, stem):
    """Follow pagination parameters only; never forward credentials to internal links."""
    params = {"page[limit]": 1000, **query}
    rows, records, seen = [], [], set()
    while True:
        key = tuple(sorted((k, str(v)) for k, v in params.items()))
        if key in seen:
            raise ValueError("OMS pagination did not advance")
        seen.add(key)
        name = f"{stem}_{len(records)+1:04d}.json" if stem == "lumisections" else f"{stem}.json"
        payload, record = save_response(session.get(OMS+endpoint, params=params, timeout=60), folder/name)
        page = payload.get("data")
        if not isinstance(page, list):
            raise ValueError("Expected an OMS data array")
        rows.extend(page); records.append(record)
        link = payload.get("links", {}).get("next")
        link = link.get("href") if isinstance(link, dict) else link
        if not link:
            total = payload.get("meta", {}).get("totalResourceCount")
            if total is not None and len(rows) != int(total):
                raise ValueError("Incomplete OMS pagination")
            return rows, records
        pagination = {k: v for k, v in parse_qsl(urlsplit(link).query) if k.startswith("page[")}
        if not page or stem != "lumisections" or not pagination:
            raise ValueError("Invalid OMS pagination")
        params = {**query, **pagination}


def oms_collect_run(run, session):
    folder = DATA/"oms/raw"/str(run)
    (folder/"source.json").unlink(missing_ok=True)
    runs, records = oms_pages(session, "runs", {"filter[run_number][EQ]": run}, folder, "runs")
    if len(runs) != 1 or runs[0]["attributes"].get("run_number") != run:
        raise ValueError("OMS run unavailable")
    rows, pages = oms_pages(session, "lumisections", {"filter[run_number][EQ]": run}, folder, "lumisections")
    records.extend(pages)
    fills = {row["attributes"].get("fill_number") for row in rows}
    for fill in sorted(number for number in fills if number is not None and number > 0):
        values, pages = oms_pages(session, "fills", {"filter[fill_number][EQ]": fill}, folder, f"fills_{fill}")
        if any(row["attributes"].get("fill_number") != fill for row in values):
            raise ValueError("OMS returned another fill")
        records.extend(pages)
    write_source(folder, records, run_number=run, units="API units unchanged")
    return oms_summary(run)


def oms_summary(run):
    _, saved = read_raw(DATA/"oms/raw"/str(run))
    run_rows = saved["runs.json"]["data"]
    if len(run_rows) != 1 or run_rows[0]["attributes"].get("run_number") != run:
        raise ValueError("OMS run does not match")
    fills = [row["attributes"] for name, data in saved.items() if name.startswith("fills_") for row in data["data"]]
    bunches = {row["fill_number"]: row.get("bunches_colliding") for row in fills}
    if len(bunches) != len(fills):
        raise ValueError("Duplicate OMS fill")
    rows = [row["attributes"] for name, data in saved.items() if name.startswith("lumisections_") for row in data["data"]]
    frame = pd.DataFrame([{key: row.get(key) for key in OMS_FIELDS} for row in rows], columns=OMS_FIELDS)
    frame = frame.rename(columns={"lumisection_number": "ls_number"})
    if (not frame.run_number.eq(run).all() or frame.ls_number.isna().any()
            or (frame.ls_number <= 0).any() or frame.ls_number.duplicated().any()):
        raise ValueError("Invalid OMS run/LS identifiers")
    for key in OMS_FIELDS:
        if key.endswith("_ready") or key in ("beams_stable", "cms_active"):
            frame[key] = frame[key].astype("boolean")
    frame["bunches_colliding"] = frame.fill_number.map(bunches)
    return write_csv(frame.sort_values("ls_number"), "oms", run)


def rr_session():
    cache = Path(os.environ.setdefault("TSGAUTH_AUTHDIR", str(Path.home()/".cache/ml4dc-rr-auth")))
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    from tsgauth.oidcauth import KerbSessionAuth
    session = KerbSessionAuth(url=RR+"/runs", use_auth_file=True).session()
    if urlsplit(session.get(RR+"/runs", timeout=30).url).hostname != urlsplit(RR).hostname:
        raise ValueError("Run Registry SSO did not reach the application")
    return session


def rr_datasets(session, filters, folder):
    (folder/"source.json").unlink(missing_ok=True)
    records, datasets, expected, page = [], [], None, 0
    while True:
        query = dict(page=page, filter=filters, page_size=100, sortings=[])
        response = session.post(RR+"/datasets_filtered_ordered", json=query, timeout=30, allow_redirects=False)
        payload, record = save_response(response, folder/f"datasets_{page+1:04d}.json")
        record.update(kind="datasets", request=query); records.append(record)
        rows, count = payload["datasets"], int(payload["count"])
        if not isinstance(rows, list) or count < 0 or (expected is not None and count != expected):
            raise ValueError("Invalid RR dataset catalogue")
        expected = count; datasets.extend(rows)
        identifiers = {(row.get("run_number"), row["name"]) for row in datasets}
        if len(identifiers) != len(datasets) or len(datasets) > expected:
            raise ValueError("Duplicate RR datasets")
        if len(datasets) == expected:
            break
        if not rows:
            raise ValueError("Incomplete RR pagination")
        page += 1
    return datasets, records


def rr_collect_run(run, session):
    folder = DATA/"rr/raw"/str(run)
    datasets, records = rr_datasets(session, {"run_number": run}, folder)
    names = [row["name"] for row in datasets if row["name"].startswith("/PromptReco/")]
    for index, name in enumerate(names):
        query = dict(run_number=run, dataset_name=name)
        response = session.post(RR+"/lumisections/rr_lumisection_ranges_by_component", json=query,
                                timeout=30, allow_redirects=False)
        _, record = save_response(response, folder/f"components_{index+1:04d}.json")
        record.update(kind="components", rr_dataset=name, request=query); records.append(record)
    write_source(folder, records, run_number=run, component="muon-muon", selection="PromptReco only")
    return rr_summary(run)


def rr_collect_year(year, session):
    folder = DATA/"rr/raw"/f"year_{year}"
    name = f"/PromptReco/Collisions{year}/DQM"
    _, records = rr_datasets(session, {"name": name}, folder)
    write_source(folder, records, rr_dataset=name, component="muon-muon")
    return rr_year_summary(year)


def rr_year_summary(year):
    """One year of official Muon run labels, including unknown verdicts."""
    _, saved = read_raw(DATA/"rr/raw"/f"year_{year}")
    rows = []
    for payload in saved.values():
        for dataset in payload["datasets"]:
            triplets = (dataset.get("DatasetTripletCache") or {}).get("triplet_summary") or {}
            counts = triplets.get("muon-muon") or {}
            good, bad = int(counts.get("GOOD", 0)), int(counts.get("BAD", 0))
            other = sum(value for key, value in counts.items()
                        if key not in ("GOOD", "BAD") and isinstance(value, (int, float)))
            label = "bad" if bad > 0 else "good" if good > 0 and other == 0 else "unknown"
            if dataset.get("deleted"):
                label = "unknown"
            rows.append(dict(run_number=dataset["run_number"], rr_dataset=dataset["name"],
                             label=label, good_ls=good, bad_ls=bad, other_ls=int(other)))
    columns = ["run_number", "rr_dataset", "label", "good_ls", "bad_ls", "other_ls"]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.run_number.duplicated().any():
        raise ValueError("Duplicate RR run verdicts")
    existing = DATA/"rr/summary/runs.csv"
    if existing.exists():
        previous = pd.read_csv(existing)
        previous = previous[previous.rr_dataset.ne(f"/PromptReco/Collisions{year}/DQM")]
        frame = pd.concat([previous[columns], frame], ignore_index=True)
    return write_csv(frame.sort_values("run_number"), "rr", "runs")


def rr_summary(run):
    source, saved = read_raw(DATA/"rr/raw"/str(run))
    rows = []
    for record in source["requests"]:
        dataset = record.get("rr_dataset", "")
        if record.get("kind") != "components" or not dataset.startswith("/PromptReco/"):
            continue
        for interval in saved[record["file"]].get("muon-muon", []):
            start, end = interval["start"], interval["end"]
            if type(start) is not int or type(end) is not int or not 0 < start <= end:
                raise ValueError("Invalid RR Muon interval")
            status = interval.get("status")
            rows.append(dict(run_number=run, rr_dataset=dataset, start=start, end=end, status=status,
                             label={"GOOD": "good", "BAD": "bad"}.get(status, "unknown"),
                             comment=interval.get("comment"), cause=interval.get("cause")))
    return write_csv(pd.DataFrame(rows, columns=RR_FIELDS), "rr", run)


def selected_runs(source, offline, years=None):
    if offline:
        return sorted(int(path.name) for path in (DATA/source/"raw").glob("*")
                      if path.is_dir() and path.name.isdigit())
    cached = None
    if MANIFEST.exists():
        frame = pd.read_csv(MANIFEST)
        frame = frame[frame.dataset.str.startswith("/Muon0/Run20")]
        if years:
            frame = frame[frame.dataset.str.contains("|".join(f"/Run{year}" for year in years))]
        cached = set(frame.run_number.astype(int))
    labels = DATA/"rr/summary/runs.csv"
    if labels.exists():
        frame = pd.read_csv(labels)
        if years:
            frame = frame[frame.rr_dataset.str.contains("|".join(f"Collisions{year}/" for year in years))]
        runs = set(frame.run_number.astype(int))
        return sorted(runs if cached is None else runs & cached)
    if cached is not None:
        return sorted(cached)
    raise ValueError("Run RR metadata first or specify --runs; no RR summary or Muon0 catalogue exists")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("dcs", "oms", "rr"))
    parser.add_argument("--years", type=int, nargs="+")
    parser.add_argument("--runs", type=int, nargs="+")
    parser.add_argument("--offline", action="store_true", help="Rebuild CSV summaries from saved raw JSON")
    args = parser.parse_args()
    if args.runs and any(run <= 0 for run in args.runs):
        parser.error("Run numbers must be positive")
    if not args.offline:
        setup_tls()
    if args.source == "dcs":
        years = args.years or [2023, 2024, 2025, 2026]
        dcs_collect(years, offline=args.offline)
        print(f"DCS: {len(years)} yearly CSV summaries saved")
        return
    if args.source == "rr" and not args.runs:
        session = None if args.offline else rr_session()
        for year in args.years or [2025]:
            (rr_year_summary(year) if args.offline else rr_collect_year(year, session))
        print("RR: yearly Muon run labels saved → data/rr/summary/runs.csv")
        return
    try:
        runs = list(dict.fromkeys(args.runs or selected_runs(args.source, args.offline, args.years)))
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not runs:
        parser.error("No runs found; specify --runs or populate the Muon0 run catalogue")
    session, auth = None, None
    if not args.offline and args.source == "oms":
        from tsgauth.oidcauth import DeviceAuth
        auth = DeviceAuth("cmsoms-prod-public", target_client_id="cmsoms-prod")
        session = requests.Session()
    elif not args.offline:
        session = rr_session()
    failures, completed = [], 0
    for run in runs:
        try:
            if args.offline:
                (oms_summary if args.source == "oms" else rr_summary)(run)
            elif args.source == "oms":
                session.headers.update(auth.authparams()["headers"])
                oms_collect_run(run, session)
            else:
                rr_collect_run(run, session)
            completed += 1
        except (requests.RequestException, ValueError, OSError, KeyError) as error:
            failures.append(dict(run_number=int(run), error=str(error)))
        if (completed + len(failures)) % 25 == 0:
            print(f"{args.source.upper()}: {completed + len(failures)}/{len(runs)} runs checked", flush=True)
    folder = DATA/args.source/"summary"; folder.mkdir(parents=True, exist_ok=True)
    if failures:
        (folder/"failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    else:
        (folder/"failures.json").unlink(missing_ok=True)
    print(f"{args.source.upper()}: {completed} runs saved, {len(failures)} skipped → {folder}")
    if not completed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
