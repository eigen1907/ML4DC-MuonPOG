"""Collect annual DCS/OMS/RR metadata and official Muon certification masks."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from urllib.parse import parse_qsl, urlsplit

import requests

from scripts.models.config import cached_runs

DATA = Path("data")
OMS = "https://cmsoms.cern.ch/agg/api/v1/"
RR = "https://cmsrunregistry.web.cern.ch/api"
OMS_FIELDS = (
    "run_number", "lumisection_number", "start_time", "end_time",
    "delivered_lumi_per_lumisection", "recorded_lumi_per_lumisection", "pileup",
    "beams_stable", "cms_active",
)
RR_COMPONENTS = {
    "cms": "cms-cms", "muon": "muon-muon", "dt": "dt-dt", "csc": "csc-csc",
    "rpc": "rpc-rpc", "gem": "gem-gem", "tracker_pixel": "tracker-pixel",
    "tracker_strip": "tracker-strip", "tracking": "tracker-track",
    "ecal": "ecal-ecal", "hcal": "hcal-hcal", "hlt": "hlt-hlt",
    "l1t": "l1t-l1t", "lumi": "lumi-lumi",
}


def setup_tls():
    if sys.platform == "darwin":
        import truststore
        truststore.inject_into_ssl()


def json_response(response):
    response.raise_for_status()
    if "json" not in response.headers.get("content-type", "").lower():
        raise ValueError("Expected JSON; check CERN authentication")
    return json.loads(response.content)


def metadata_path(source, year):
    return DATA/source/f"{year}.json"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def dcs_url(year):
    base = "https://cms-service-dqmdc.web.cern.ch/CAF/certification/"
    if year == 2023:
        return base + "Collisions23/DCSOnly_JSONS/Collisions23_13p6TeV_eraBCD_366403_370790_DCSOnly_TkPx.json"
    if year not in (2024, 2025, 2026):
        raise ValueError("Supported DCS snapshots: 2023–2026")
    short = str(year)[2:]
    return base + f"Collisions{short}/DCSOnly_JSONS/dailyDCSOnlyJSON/Collisions{short}_13p6TeV_Latest.json"


def dcs_collect(years, *, session=None):
    """Public tracker-ready intervals; preserve downloaded JSON bytes."""
    session = session or requests.Session()
    for year in years:
        path = metadata_path("dcs", year)
        response = session.get(dcs_url(year), timeout=60)
        payload = json_response(response)
        if not isinstance(payload, dict) or not payload:
            raise ValueError("Expected nonempty DCS run/LS intervals")
        for run, ranges in payload.items():
            if not str(run).isdigit() or int(run) <= 0 or not isinstance(ranges, list):
                raise ValueError("Invalid DCS run")
            for pair in ranges:
                if (not isinstance(pair, list) or len(pair) != 2
                        or any(type(x) is not int for x in pair) or not 0 < pair[0] <= pair[1]):
                    raise ValueError("Invalid inclusive DCS LS range")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)


def validate_lumi_mask(payload):
    """Validate the standard inclusive CMS run-to-LS-interval JSON format."""
    if not isinstance(payload, dict) or not payload:
        raise ValueError('Expected a nonempty CMS run/LS certification JSON')
    for run, ranges in payload.items():
        if not str(run).isdigit() or int(run) <= 0 or not isinstance(ranges, list):
            raise ValueError('Invalid certification run')
        for pair in ranges:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(x) is not int for x in pair) or not 0 < pair[0] <= pair[1]):
                raise ValueError('Invalid inclusive certification LS range')


def muon_collect(years, *, session=None):
    """Save official Muon masks verbatim, with separate 2025/2026 low-PU masks.

    Exact case-insensitive filename matching excludes per-era, pp-reference
    and before-review alternatives. Choose the largest ending run, not the
    directory's incidental ordering. URLs and byte hashes are printed so a
    collected snapshot can be identified without another manifest file.
    """
    session = session or requests.Session()
    for year in years:
        if year not in (2024, 2025, 2026):
            raise ValueError('Supported Muon certification years: 2024–2026')
        base = f'https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions{str(year)[-2:]}/'
        listing = session.get(base, timeout=60)
        listing.raise_for_status()
        names = {link.rsplit('/', 1)[-1] for link in
                 re.findall(r'href=[\"\']([^\"\']+)[\"\']', listing.text, re.IGNORECASE)}
        pattern = re.compile(rf'Cert_Collisions{year}_(\d+)_(\d+)_muon\.json', re.IGNORECASE)
        candidates = [(int(match[2]), int(match[1]), name) for name in names
                      if (match := pattern.fullmatch(name))]
        if not candidates:
            raise ValueError(f'No official Muon certification JSON found for {year}')
        targets = [(max(candidates)[2], DATA/'rr'/f'{year}_muon.json')]
        if year == 2025:
            low_pu = next((name for preferred in ('2025_lowpu_updated.json', '2025_lowpu.json')
                           for name in names if name.lower() == preferred), None)
            if low_pu is None:
                raise ValueError('No official 2025 low-PU certification JSON found')
            targets.append((low_pu, DATA/'rr'/'2025_lowPU.json'))
        if year == 2026:
            low_pu = [name for name in names if name.lower() == f'cert_collisions{year}_lowpu.json']
            if len(low_pu) != 1:
                raise ValueError('Expected one official 2026 low-PU certification JSON')
            targets.append((low_pu[0], DATA/'rr'/'2026_lowPU.json'))
        for name, path in targets:
            response = session.get(base + name, timeout=60)
            if (year == 2025 and name.lower() == '2025_lowpu_updated.json'
                    and response.status_code == 404):
                fallback = next((item for item in names if item.lower() == '2025_lowpu.json'), None)
                if fallback is not None:
                    name = fallback
                    response = session.get(base + name, timeout=60)
            payload = json_response(response)
            validate_lumi_mask(payload)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(response.content)
            digest = hashlib.sha256(response.content).hexdigest()
            print(f'{path}: {base + name}\n  sha256 {digest}')


def oms_pages(session, endpoint, query):
    """Read all pages, retaining only requested LS attributes."""
    params, selected, seen = {"page[limit]": 1000, **query}, [], set()
    while True:
        key = tuple(sorted((name, str(value)) for name, value in params.items()))
        if key in seen:
            raise ValueError("OMS pagination did not advance")
        seen.add(key)
        payload = json_response(session.get(OMS+endpoint, params=params, timeout=60))
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise ValueError("Expected an OMS data array")
        selected.extend({name: row["attributes"].get(name) for name in OMS_FIELDS} for row in rows)
        link = payload.get("links", {}).get("next")
        link = link.get("href") if isinstance(link, dict) else link
        if not link:
            total = payload.get("meta", {}).get("totalResourceCount")
            if total is not None and len(selected) != int(total):
                raise ValueError("Incomplete OMS pagination")
            return selected
        pagination = {name: value for name, value in parse_qsl(urlsplit(link).query)
                      if name.startswith("page[")}
        if not rows or endpoint != "lumisections" or not pagination:
            raise ValueError("Invalid OMS pagination")
        params = {**query, **pagination}


def validate_oms(rows):
    identifiers = [(row.get("run_number"), row.get("lumisection_number")) for row in rows]
    if (any(type(value) is not int or value <= 0 for pair in identifiers for value in pair)
            or len(set(identifiers)) != len(identifiers)):
        raise ValueError("Invalid or duplicate OMS run/LS identifiers")


def oms_collect_run(run, session, years=None, cache=None):
    """Update one run; an optional batch cache defers annual file writes."""
    rows = oms_pages(session, "lumisections", {
        "filter[run_number][EQ]": run, "fields": ",".join(OMS_FIELDS)})
    if not rows or any(row.get("run_number") != run for row in rows):
        raise ValueError("OMS run unavailable or returned another run")
    try:
        observed_years = {datetime.fromisoformat(row["start_time"].replace("Z", "+00:00")).year for row in rows}
    except (KeyError, TypeError, ValueError, AttributeError) as error:
        raise ValueError("OMS LS timestamps are needed to select the annual file") from error
    if len(observed_years) != 1:
        raise ValueError("OMS run spans multiple years; no annual file was changed")
    year = observed_years.pop()
    if years and year not in years:
        raise ValueError("OMS run does not belong to the requested year")
    validate_oms(rows)
    path = metadata_path("oms", year)
    previous = cache.get(year) if cache is not None else None
    if previous is None:
        previous = json.loads(path.read_text()) if path.exists() else dict(lumisections=[])
    raw = {"lumisections": [row for row in previous["lumisections"]
                            if row.get("run_number") != run] + rows}
    if cache is not None:
        cache[year] = raw
        return raw
    validate_oms(raw["lumisections"])
    raw["lumisections"].sort(key=lambda row: (row["run_number"], row["lumisection_number"]))
    write_json(path, raw)
    return raw


def oms_collect_runs(runs, session, years=None, auth=None):
    """Collect a batch in memory and save each annual JSON once."""
    cache, saved, failed = {}, 0, 0
    for run in runs:
        try:
            if auth is not None:
                session.headers.update(auth.authparams()["headers"])
            oms_collect_run(run, session, years, cache)
            saved += 1
        except (requests.RequestException, ValueError, OSError, KeyError) as error:
            failed += 1
            print(f"OMS run {run} skipped: {error}", file=sys.stderr)
    for year, raw in cache.items():
        validate_oms(raw["lumisections"])
        raw["lumisections"].sort(key=lambda row: (row["run_number"], row["lumisection_number"]))
        write_json(metadata_path("oms", year), raw)
    return saved, failed


def rr_session():
    cache = Path(os.environ.setdefault("TSGAUTH_AUTHDIR", str(Path.home()/".cache/ml4dc-rr-auth")))
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    from tsgauth.oidcauth import KerbSessionAuth
    session = KerbSessionAuth(url=RR+"/runs", use_auth_file=True).session()
    if urlsplit(session.get(RR+"/runs", timeout=30).url).hostname != urlsplit(RR).hostname:
        raise ValueError("Run Registry SSO did not reach the application")
    return session


def rr_datasets(session, name, runs=None):
    """Read one RR dataset selection without keeping API wrappers."""
    if runs is not None:
        runs = list(runs)
        if not runs:
            return []
    records, seen, expected, page = [], set(), None, 0
    filters = {"name": name}
    if runs is not None:
        filters["run_number"] = list(runs)
    while True:
        query = dict(page=page, filter=filters, page_size=100, sortings=[])
        payload = json_response(session.post(RR+"/datasets_filtered_ordered", json=query,
                                             timeout=30, allow_redirects=False))
        rows, count = payload["datasets"], int(payload["count"])
        if not isinstance(rows, list) or count < 0 or (expected is not None and count != expected):
            raise ValueError("Invalid RR dataset catalogue")
        expected = count
        for row in rows:
            run = row["run_number"]
            if (row["name"] != name or type(run) is not int or run <= 0 or run in seen
                    or (runs is not None and run not in runs)):
                raise ValueError("Duplicate or unexpected RR dataset")
            seen.add(run)
            triplets = (row.get("DatasetTripletCache") or {}).get("triplet_summary") or {}
            records.append({"run_number": run, "rr_dataset": name, "deleted": bool(row.get("deleted")),
                            "components": {short: {key: value for key, value in
                                (triplets.get(component) or {}).items() if isinstance(value, (int, float))}
                                for short, component in RR_COMPONENTS.items()}})
        if len(seen) == expected:
            return records
        if len(seen) > expected or not rows:
            raise ValueError("Incomplete RR pagination")
        page += 1


def rr_collect_year(year, session, runs=None):
    """Keep Collisions verdicts, then cover cached Muon0 with bounded fallbacks."""
    if runs is None:
        runs = [row["run_number"] for row in cached_runs()
                if row["dataset"].startswith(f"/Muon0/Run{year}")]
    runs = list(runs)
    if any(type(run) is not int or run <= 0 for run in runs):
        raise ValueError("Expected positive integer Muon0 run numbers")
    targets = sorted(set(runs))
    records = {row["run_number"]: row for row in
               rr_datasets(session, f"/PromptReco/Collisions{year}/DQM")}
    for name in (f"/PromptReco/Commissioning{year}/DQM", "online"):
        missing = [run for run in targets if run not in records or records[run]["deleted"]]
        for first in range(0, len(missing), 100):
            for row in rr_datasets(session, name, missing[first:first+100]):
                previous = records.get(row["run_number"])
                if previous is None or (previous["deleted"] and not row["deleted"]):
                    records[row["run_number"]] = row
    value = {"runs": sorted(records.values(), key=lambda record: record["run_number"])}
    write_json(metadata_path("rr", year), value)
    return value


def selected_runs(years=None):
    cached = {row["run_number"] for row in cached_runs()
              if row["dataset"].startswith("/Muon0/Run20")
              and (not years or any(f"/Run{year}" in row["dataset"] for year in years))}
    if cached:
        return sorted(cached)
    paths = ([metadata_path("rr", year) for year in years] if years else
             sorted((DATA/"rr").glob("[0-9][0-9][0-9][0-9].json")))
    runs = {int(run["run_number"]) for path in paths if path.exists()
            for run in json.loads(path.read_text())["runs"]}
    if runs:
        return sorted(runs)
    raise ValueError("Run RR metadata first or specify --runs; no Muon0 run list is available")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=("dcs", "oms", "rr", "muon"))
    parser.add_argument("--years", type=int, nargs="+")
    parser.add_argument("--runs", type=int, nargs="+", help="OMS run selection")
    args = parser.parse_args()
    if args.runs and args.source != "oms":
        parser.error("--runs is for OMS collection; use --years for annual files")
    if args.runs and any(run <= 0 for run in args.runs):
        parser.error("Run numbers must be positive")
    setup_tls()
    if args.source == "dcs":
        dcs_collect(args.years or [2023, 2024, 2025, 2026])
    elif args.source == "muon":
        muon_collect(args.years or [2024, 2025, 2026])
    elif args.source == "rr":
        session = rr_session()
        for year in args.years or [2025]:
            rr_collect_year(year, session)
    else:
        try:
            runs = list(dict.fromkeys(args.runs or selected_runs(args.years)))
        except (ValueError, OSError) as error:
            parser.error(str(error))
        from tsgauth.oidcauth import DeviceAuth
        auth = DeviceAuth("cmsoms-prod-public", target_client_id="cmsoms-prod")
        saved, failed = oms_collect_runs(runs, requests.Session(), args.years, auth)
        print(f"OMS: {saved} runs saved, {failed} skipped")
        if not saved:
            raise SystemExit(1)
        return
    print(f"{args.source.upper()}: yearly JSON files saved")


if __name__ == "__main__":
    main()
