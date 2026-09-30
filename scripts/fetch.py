"""Fetch one Muons/ histogram for one DIALS dataset and run."""

import argparse
from pathlib import Path

from cmsdials import Dials
from cmsdials.auth.bearer import Credentials
from cmsdials.filters import LumisectionHistogram1DFilters


def fetch_one(dials: Dials, dataset: str, run: int, me: str) -> bool | None:
    """Return True if saved, False if cached, or None if no rows exist."""
    output = (
        Path("data/raw")
        / dataset.strip("/").replace("/", "__")
        / str(run)
        / (me.replace("/", "__") + ".parquet")
    )
    if output.exists():
        return False

    result = dials.h1d.list_all(
        LumisectionHistogram1DFilters(
            dataset=dataset, run_number=run, me=me, page_size=100
        ),
        retries=3,
    )
    if result.next is not None or result.exc_type is not None:
        raise RuntimeError(f"Incomplete DIALS query: {dataset}, run {run}, {me}")
    if not result.results:
        print(f"No rows: {dataset}, run {run}, {me}", flush=True)
        return None

    frame = result.to_pandas().sort_values("ls_number")
    if not (
        frame.dataset.eq(dataset).all()
        and frame.run_number.eq(run).all()
        and frame.me.eq(me).all()
    ):
        raise RuntimeError(f"Unexpected DIALS rows: {dataset}, run {run}, {me}")
    if frame.ls_number.duplicated().any():
        raise RuntimeError(f"Duplicate lumisections: {dataset}, run {run}, {me}")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".parquet.tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(output)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--run", required=True, type=int)
    parser.add_argument("--me", required=True)
    args = parser.parse_args()
    if not args.me.startswith("Muons/"):
        parser.error("--me must be a Muons/ path")

    dials = Dials(Credentials.from_creds_file(), workspace="muo")
    saved = fetch_one(dials, args.dataset, args.run, args.me)
    if saved is not None:
        print("Saved" if saved else "Already cached", args.dataset, args.run, args.me)


if __name__ == "__main__":
    main()
