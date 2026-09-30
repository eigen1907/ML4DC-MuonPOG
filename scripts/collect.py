"""Download 2025 Muon0 data, excluding non-PromptReco 2025G test datasets."""

import csv
from pathlib import Path

from cmsdials import Dials
from cmsdials.auth.bearer import Credentials
from cmsdials.filters import RunFilters

from fetch import fetch_one


def main() -> None:
    dials = Dials(Credentials.from_creds_file(), workspace="muo")
    response = dials.run.list_all(
        RunFilters(dataset__regex="Muon0/Run2025", page_size=100),
        retries=3,
    )
    if response.next is not None or response.exc_type is not None:
        raise RuntimeError("Incomplete 2025 Muon0 run listing")
    runs = sorted(
        (
            row for row in response.results
            if row.dataset.startswith("/Muon0/Run2025")
            and row.dataset.endswith("/DQMIO")
            and (
                not row.dataset.startswith("/Muon0/Run2025G-")
                or row.dataset == "/Muon0/Run2025G-PromptReco-v1/DQMIO"
            )
        ),
        key=lambda row: (row.dataset, row.run_number),
    )
    mes = sorted(me.me for me in dials.mes.list() if me.me.startswith("Muons/") and me.dim == 1)
    if not runs or not mes:
        raise RuntimeError("No matching runs or Muons/ 1D MEs")

    Path("data").mkdir(exist_ok=True)
    with Path("data/runs.csv").open("w", newline="") as output:
        writer = csv.writer(output)
        writer.writerow(["dataset", "run_number", "ls_count", "ls_completeness"])
        writer.writerows(
            (row.dataset, row.run_number, row.ls_count, row.ls_completeness)
            for row in runs
        )

    print(f"Found {len(runs)} dataset/run pairs and {len(mes)} Muons/ MEs", flush=True)
    for index, row in enumerate(runs, 1):
        print(f"{index}/{len(runs)}  {row.dataset}  run {row.run_number}", flush=True)
        status = [fetch_one(dials, row.dataset, row.run_number, me) for me in mes]
        print(f"  {status.count(True)} new, {status.count(None)} empty", flush=True)


if __name__ == "__main__":
    main()
