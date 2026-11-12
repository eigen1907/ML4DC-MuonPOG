"""Save Run Registry responses and extract Muon certification intervals."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from metadata import save_response, setup_tls, write_source


ROOT = Path("data/rr")
API = "https://cmsrunregistry.web.cern.ch/api"
PAGE_SIZE = 100
COLUMNS = ["run_number", "rr_dataset", "start", "end", "status", "comment", "cause"]


def authenticated_session():
    """Use personal Kerberos/OTP SSO; keep its cache outside the repository."""
    cache = Path(os.environ.setdefault("TSGAUTH_AUTHDIR", str(Path.home()/".cache/ml4dc-rr-auth")))
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    setup_tls()
    from tsgauth.oidcauth import KerbSessionAuth

    session = KerbSessionAuth(url=API + "/runs", use_auth_file=True).session()
    # Complete the application callback before sending a POST query. The GET
    # route itself may return 404/405; a CERN login page means SSO is incomplete.
    response = session.get(API + "/runs", timeout=30)
    if urlparse(response.url).hostname != urlparse(API).hostname:
        raise RuntimeError("Run Registry SSO did not reach the application")
    return session


def collect_run(run, session):
    """Download all dataset pages and all component responses for one run."""
    folder = ROOT/"raw"/str(run)
    (folder/"source.json").unlink(missing_ok=True)
    records, names, expected, page = [], [], None, 0
    while True:
        query = {"page": page, "filter": {"run_number": run},
                 "page_size": PAGE_SIZE, "sortings": []}
        response = session.post(API + "/datasets_filtered_ordered", json=query,
                                timeout=30, allow_redirects=False)
        payload, record = save_response(response, folder/f"datasets_{page + 1:04d}.json")
        record.update(kind="datasets", request=query)
        records.append(record)
        rows, count = payload["datasets"], int(payload["count"])
        if not isinstance(rows, list) or count < 0 or (expected is not None and count != expected):
            raise ValueError("Run Registry dataset catalogue changed or is invalid")
        expected = count
        names.extend(row["name"] for row in rows)
        if len(set(names)) != len(names) or len(names) > expected:
            raise ValueError("Duplicate or excess Run Registry datasets")
        if len(names) == expected:
            break
        if not rows:
            raise ValueError("Run Registry dataset pagination ended early")
        page += 1

    for index, name in enumerate(names, 1):
        query = {"run_number": run, "dataset_name": name}
        response = session.post(API + "/lumisections/rr_lumisection_ranges_by_component",
                                json=query, timeout=30, allow_redirects=False)
        _, record = save_response(response, folder/f"components_{index:04d}.json")
        record.update(kind="components", rr_dataset=name, request=query)
        records.append(record)
    write_source(folder, records, run_number=run, service=API,
                 authentication="Personal Kerberos + OTP SSO",
                 component="muon-muon", bounds="Original RR start/end; intervals are not expanded")


def process_run(run):
    """Extract Muon intervals; preserve all statuses and separate RR datasets."""
    folder = ROOT/"raw"/str(run)
    source = json.loads((folder/"source.json").read_text())
    rows = []
    for record in source["requests"]:
        body = (folder/record["file"]).read_bytes()
        if hashlib.sha256(body).hexdigest() != record["sha256"]:
            raise ValueError("Saved Run Registry response differs from its provenance")
        if record.get("kind") != "components":
            continue
        payload = json.loads(body)
        for interval in payload.get("muon-muon", []):
            start, end = interval["start"], interval["end"]
            if type(start) is not int or type(end) is not int or not 0 < start <= end:
                raise ValueError("Invalid Run Registry Muon interval")
            rows.append(dict(run_number=run, rr_dataset=record["rr_dataset"],
                             start=start, end=end, status=interval.get("status"),
                             comment=interval.get("comment"), cause=interval.get("cause")))
    frame = pd.DataFrame(rows, columns=COLUMNS).astype(
        {"run_number": "int64", "start": "int64", "end": "int64",
         **{name: "string" for name in ("rr_dataset", "status", "comment", "cause")}})
    path = ROOT/"processed"/f"{run}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, nargs="+", required=True)
    parser.add_argument("--offline", action="store_true", help="Rebuild tables from saved raw responses")
    args = parser.parse_args()
    if any(run <= 0 for run in args.runs):
        parser.error("Run numbers must be positive")
    session = None if args.offline else authenticated_session()
    for run in dict.fromkeys(args.runs):
        if session is not None:
            collect_run(run, session)
        frame = process_run(run)
        print(f"RR {run}: {len(frame)} Muon intervals saved")


if __name__ == "__main__":
    main()
