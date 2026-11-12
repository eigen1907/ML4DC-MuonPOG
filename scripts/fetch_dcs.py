"""Cache public tracker DCS JSON unchanged and extract inclusive LS intervals."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from metadata import save_response, setup_tls


ROOT = Path("data/dcs")
SEMANTICS = "Inclusive tracker DCS-only intervals; not Muon certification labels"


def source_url(year):
    base = "https://cms-service-dqmdc.web.cern.ch/CAF/certification/"
    if year == 2023:
        return base + "Collisions23/DCSOnly_JSONS/Collisions23_13p6TeV_eraBCD_366403_370790_DCSOnly_TkPx.json"
    if year not in (2024, 2025, 2026):
        raise ValueError("Supported DCS snapshots: 2023–2026")
    short = str(year)[2:]
    return base + f"Collisions{short}/DCSOnly_JSONS/dailyDCSOnlyJSON/Collisions{short}_13p6TeV_Latest.json"


def intervals(payload):
    """Keep supplied range boundaries without expanding or inventing LS labels."""
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Expected a nonempty run-to-DCS-interval JSON")
    rows = []
    for run, ranges in payload.items():
        if not str(run).isdigit() or int(run) <= 0 or not isinstance(ranges, list):
            raise ValueError("Invalid DCS run or interval list")
        for pair in ranges:
            if (not isinstance(pair, list) or len(pair) != 2
                    or any(type(value) is not int for value in pair)
                    or not 0 < pair[0] <= pair[1]):
                raise ValueError("Expected positive inclusive DCS LS boundaries")
            rows.append((int(run), pair[0], pair[1]))
    return pd.DataFrame(rows, columns=["run_number", "first_ls", "last_ls"]).astype("int64")


def collect(years, *, offline=False, session=None):
    if not offline and session is None:
        import requests
        session = requests.Session()
    for year in years:
        raw = ROOT / "raw" / f"{year}.json"
        if offline:
            body = raw.read_bytes()
            source = json.loads(raw.with_suffix(".source.json").read_text())
            if hashlib.sha256(body).hexdigest() != source["sha256"]:
                raise ValueError("Saved DCS response differs from its provenance")
            payload = json.loads(body)
        else:
            response = session.get(source_url(year), timeout=60)
            payload, source = save_response(response, raw)
            raw.with_suffix(".source.json").write_text(json.dumps(
                {**source, "year": year, "semantics": SEMANTICS}, indent=2) + "\n")
        frame = intervals(payload)
        output = ROOT / "processed" / f"{year}.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output, index=False)
        print(f"DCS {year}: {len(frame):,} intervals → {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025, 2026])
    parser.add_argument("--offline", action="store_true", help="Rebuild processed tables from cached raw JSON")
    args = parser.parse_args()
    if not args.offline:
        setup_tls()
    collect(args.years, offline=args.offline)


if __name__ == "__main__":
    main()
