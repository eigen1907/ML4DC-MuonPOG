"""Save complete OMS responses and extract one small LS table per requested run."""

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import pandas as pd
import requests

from metadata import save_response, setup_tls, write_source


ROOT = Path("data/oms")
API = "https://cmsoms.cern.ch/agg/api/v1/"
LS_FIELDS = (
    "run_number", "lumisection_number", "fill_number", "start_time", "end_time",
    "delivered_lumi_per_lumisection", "recorded_lumi_per_lumisection", "pileup",
    "cms_active", "beams_stable", "physics_flag", "prescale_index", "prescale_name",
    "dt0_ready", "dtm_ready", "dtp_ready", "rpc_ready", "cscm_ready", "cscp_ready",
    "gem_ready", "gemm_ready", "gemp_ready", "bpix_ready", "fpix_ready",
    "tibtid_ready", "tob_ready", "tecm_ready", "tecp_ready",
)
FILL_FIELDS = ("bunches_colliding", "bunches_beam1", "bunches_beam2", "injection_scheme")
BOOL_FIELDS = tuple(name for name in LS_FIELDS if name.endswith("_ready")) + (
    "cms_active", "beams_stable", "physics_flag")


def fetch_pages(session, endpoint, query, folder, stem):
    """Keep filters fixed and use only pagination parameters from OMS links.

    OMS sometimes returns malformed/internal next URLs. Never send credentials
    to those hosts; every request goes to the same public API endpoint.
    """
    params = {"page[limit]": 1000, **query}
    rows, records, seen = [], [], set()
    while True:
        key = tuple(sorted((name, str(value)) for name, value in params.items()))
        if key in seen:
            raise ValueError("OMS pagination did not advance")
        seen.add(key)
        response = session.get(API + endpoint, params=params, timeout=60)
        filename = f"{stem}_{len(records)+1:04d}.json" if stem == "lumisections" else f"{stem}.json"
        payload, record = save_response(response, folder/filename)
        page = payload.get("data")
        if not isinstance(page, list):
            raise ValueError("Expected an OMS JSON-API data array")
        rows.extend(page)
        records.append(record)
        next_link = payload.get("links", {}).get("next")
        if isinstance(next_link, dict):
            next_link = next_link.get("href")
        total = payload.get("meta", {}).get("totalResourceCount")
        if not next_link:
            if total is not None and len(rows) != int(total):
                raise ValueError("Incomplete OMS pagination: totalResourceCount differs")
            break
        if not page or stem != "lumisections":
            raise ValueError("Unexpected empty or non-LS OMS pagination")
        pagination = {name: value for name, value in parse_qsl(urlsplit(next_link).query)
                      if name.startswith("page[")}
        if not pagination:
            raise ValueError("OMS next link has no pagination parameters")
        params = {**query, **pagination}
    return rows, records


def process_run(run, root=ROOT):
    """Extract fields only; preserve nulls, API units and individual DCS bits."""
    folder = root/"raw"/str(run)
    source = json.loads((folder/"source.json").read_text())
    if not source.get("complete"):
        raise ValueError("OMS raw download is incomplete; fetch the run again")
    saved = {}
    for record in source["requests"]:
        name = record["file"]
        if Path(name).name != name or name in saved:
            raise ValueError("Invalid or duplicate OMS raw response filename")
        body = (folder/name).read_bytes()
        if hashlib.sha256(body).hexdigest() != record["sha256"]:
            raise ValueError(f"OMS raw response changed: {name}")
        saved[name] = json.loads(body)
    run_rows = saved["runs.json"]["data"]
    if len(run_rows) != 1 or run_rows[0]["attributes"].get("run_number") != run:
        raise ValueError("Expected exactly the requested OMS run")
    run_info = run_rows[0]["attributes"]
    fill_rows = [row["attributes"] for name, payload in saved.items() if name.startswith("fills_")
                 for row in payload["data"]]
    fills = {row["fill_number"]: row for row in fill_rows}
    if len(fills) != len(fill_rows):
        raise ValueError("Duplicate OMS fill metadata")
    rows = [row for name, payload in saved.items() if name.startswith("lumisections_")
            for row in payload["data"]]
    values = [{field: row["attributes"].get(field) for field in LS_FIELDS} for row in rows]
    table = pd.DataFrame(values, columns=LS_FIELDS).rename(columns={"lumisection_number": "ls_number"})
    if (not table.run_number.eq(run).all() or table.ls_number.isna().any()
            or (table.ls_number <= 0).any() or table.ls_number.duplicated().any()):
        raise ValueError("OMS LS identifiers are missing, duplicated or from another run")
    for field in BOOL_FIELDS:
        table[field] = table[field].astype("boolean")
    table["era"] = run_info.get("era")
    # OMS cmssw_version describes online operation, not offline PromptReco provenance.
    table["online_cmssw_version"] = run_info.get("cmssw_version")
    table["hlt_key"] = run_info.get("hlt_key")
    # Retain both observations: a run header and its LS records can name different fills.
    table["run_fill_number"] = run_info.get("fill_number")
    for field in FILL_FIELDS:
        table[field] = table.fill_number.map({number: info.get(field) for number, info in fills.items()})
    table = table.sort_values("ls_number").reset_index(drop=True)
    destination = root/"processed"/f"{run}.parquet"
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(destination, index=False)
    return table


def fetch_run(run, session, root=ROOT):
    folder = root/"raw"/str(run)
    folder.mkdir(parents=True, exist_ok=True)
    (folder/"source.json").unlink(missing_ok=True)
    runs, records = fetch_pages(session, "runs", {"filter[run_number][EQ]": run}, folder, "runs")
    if len(runs) != 1 or runs[0]["attributes"].get("run_number") != run:
        raise ValueError("OMS did not return exactly the requested run")
    rows, ls_records = fetch_pages(session, "lumisections", {"filter[run_number][EQ]": run}, folder, "lumisections")
    records.extend(ls_records)
    fill_numbers = {runs[0]["attributes"].get("fill_number")}
    fill_numbers.update(row["attributes"].get("fill_number") for row in rows)
    for fill in sorted(number for number in fill_numbers if number is not None and number > 0):
        fills, fill_records = fetch_pages(session, "fills", {"filter[fill_number][EQ]": fill}, folder, f"fills_{fill}")
        if any(row["attributes"].get("fill_number") != fill for row in fills):
            raise ValueError("OMS returned another fill than requested")
        records.extend(fill_records)
    current = {record["file"] for record in records}
    for pattern in ("lumisections_*.json", "fills_*.json"):
        for path in folder.glob(pattern):
            if path.name not in current:
                path.unlink()
    units = {name: detail["units"] for row in rows for name, detail in row.get("meta", {}).get("row", {}).items()
             if name in LS_FIELDS and "units" in detail}
    write_source(folder, records, complete=True, run_number=run,
                 selection="All returned OMS lumisections for the requested run", units=units,
                 processing="Selected fields only; no unit conversion, cuts or combined DCS flag")
    return process_run(run, root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, nargs="+", required=True)
    parser.add_argument("--offline", action="store_true", help="Rebuild processed tables from saved raw responses")
    args = parser.parse_args()
    if any(run <= 0 for run in args.runs):
        parser.error("Run numbers must be positive")
    session, auth = None, None
    if not args.offline:
        setup_tls()
        from tsgauth.oidcauth import DeviceAuth
        session = requests.Session()
        auth = DeviceAuth("cmsoms-prod-public", target_client_id="cmsoms-prod")
    for run in dict.fromkeys(args.runs):
        if auth is not None:
            session.headers.update(auth.authparams()["headers"])
        table = process_run(run) if args.offline else fetch_run(run, session)
        print(f"OMS run {run}: {len(table):,} LS → {ROOT/'processed'/f'{run}.parquet'}", flush=True)


if __name__ == "__main__":
    main()
