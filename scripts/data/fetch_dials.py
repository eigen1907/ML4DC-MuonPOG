"""Cache available Muon0 per-LS Muons/ 1D histograms from DIALS."""

import argparse
from pathlib import Path
import sys

import pyarrow.parquet as pq

from scripts.models.config import DIALS, dials_folder


def dials_client():
    if sys.platform == "darwin":
        import truststore
        truststore.inject_into_ssl()
    from cmsdials import Dials
    from cmsdials.auth.bearer import Credentials

    cache = Path.home() / ".cache/ml4dc-dials"
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    credentials = Credentials.from_creds_file(cache_dir=str(cache))
    (cache / "creds").chmod(0o600)
    return Dials(credentials, workspace="muo")


def list_runs(client, years):
    """List all processing versions; return dataset/run pairs in stable order."""
    from cmsdials.filters import RunFilters

    years = sorted(set(years))
    if not years:
        raise ValueError("Specify at least one year")
    pattern = "^/Muon0/Run(" + "|".join(str(year) for year in years) + ")"
    response = client.run.list_all(RunFilters(dataset__regex=pattern, page_size=100), retries=2)
    if response.next is not None or response.exc_type is not None:
        raise RuntimeError("Incomplete DIALS run listing")
    prefixes = tuple(f"/Muon0/Run{year}" for year in years)
    runs = [row for row in response.results
            if row.dataset.startswith(prefixes) and row.dataset.endswith("/DQMIO")]
    identifiers = [(row.dataset, row.run_number) for row in runs]
    if len(set(identifiers)) != len(identifiers):
        raise RuntimeError("Duplicate DIALS dataset/run pairs")
    return sorted(runs, key=lambda row: (row.dataset, row.run_number))


def list_mes(client):
    """Use the Muons/ 1D ME names actually published in this workspace."""
    from cmsdials.filters import MEFilters

    rows = client.mes.list(MEFilters(me__regex="^Muons/", dim=1), retries=2)
    names = [row.me for row in rows if row.me.startswith("Muons/") and row.dim == 1]
    if not names or len(set(names)) != len(names):
        raise RuntimeError("Expected unique Muons/ 1D ME names")
    return sorted(names)


def _validate(frame, row, me):
    if (frame.empty or not frame.dataset.eq(row.dataset).all()
            or not frame.run_number.eq(row.run_number).all() or not frame.me.eq(me).all()):
        raise RuntimeError(f"Unexpected DIALS rows: {row.dataset}, run {row.run_number}, {me}")
    ls = frame.ls_number
    if ls.isna().any() or ls.le(0).any() or ls.mod(1).ne(0).any() or ls.duplicated().any():
        raise RuntimeError(f"Invalid or duplicate LS: {row.dataset}, run {row.run_number}, {me}")


def fetch_run(client, row, mes=None, output=DIALS):
    """Save one complete API query per ME; retain existing valid cache files."""
    from cmsdials.filters import LumisectionHistogram1DFilters

    mes = list_mes(client) if mes is None else mes
    counts = dict(saved=0, cached=0, empty=0)
    for me in mes:
        if not me.startswith("Muons/"):
            raise ValueError("Only Muons/ MEs are supported")
        path = dials_folder(row.dataset, row.run_number, root=output) / (me.replace("/", "__") + ".parquet")
        if path.exists():
            required = {"dataset", "run_number", "ls_number", "me", "entries", "data", "x_min", "x_max", "x_bin"}
            if not required.issubset(pq.read_schema(path).names):
                raise RuntimeError(f"Incomplete cached Parquet schema: {path}")
            cached = pq.read_table(path, columns=["dataset", "run_number", "ls_number", "me"]).to_pandas()
            _validate(cached, row, me)
            counts["cached"] += 1
            continue
        response = client.h1d.list_all(LumisectionHistogram1DFilters(
            dataset=row.dataset, run_number=row.run_number, me=me, page_size=100), retries=2)
        if response.next is not None or response.exc_type is not None:
            raise RuntimeError(f"Incomplete DIALS query: {row.dataset}, run {row.run_number}, {me}")
        if not response.results:
            counts["empty"] += 1
            continue
        frame = response.to_pandas().sort_values("ls_number")
        _validate(frame, row, me)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
        counts["saved"] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--run", type=int, help="Fetch one run across its available processing versions")
    parser.add_argument("--list-only", action="store_true", help="Show availability without saving histograms")
    parser.add_argument("--output", type=Path, default=DIALS, help="Parquet cache root")
    args = parser.parse_args()
    if any(not 2000 <= year <= 2099 for year in args.years):
        parser.error("Use four-digit Run years")
    if args.run is not None and args.run <= 0:
        parser.error("Run number must be positive")
    client = dials_client()
    runs = list_runs(client, args.years)
    if args.run is not None:
        runs = [row for row in runs if row.run_number == args.run]
    mes = list_mes(client)
    for year in sorted(set(args.years)):
        rows = [row for row in runs if row.dataset.startswith(f"/Muon0/Run{year}")]
        print(f"{year}: {len({row.dataset for row in rows})} datasets, {len(rows)} dataset/run pairs", flush=True)
    print(f"Muons/ 1D MEs: {len(mes)}", flush=True)
    if args.list_only:
        return
    if not runs:
        raise SystemExit("No matching Muon0 runs in DIALS")
    for index, row in enumerate(runs, 1):
        print(f"{index}/{len(runs)} {row.dataset} run {row.run_number}", flush=True)
        counts = fetch_run(client, row, mes, args.output)
        print(f"  {counts['saved']} saved, {counts['cached']} cached, {counts['empty']} empty", flush=True)


if __name__ == "__main__":
    main()
