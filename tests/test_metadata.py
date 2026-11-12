"""Small source-response tests; no CERN login, downloads or model training."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import fetch_metadata as metadata


def response(value=None, *, body=None, status=200, kind="application/json"):
    result = requests.Response()
    result.status_code, result.url = status, metadata.OMS+"test"
    result.headers["Content-Type"] = kind
    result._content = body if body is not None else json.dumps(value).encode()
    return result


def oms_page(rows, *, total=None, next_link=None):
    return response({"data": [{"attributes": row} for row in rows],
                     "meta": {"totalResourceCount": len(rows) if total is None else total},
                     "links": {"next": next_link}})


class MetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = patch.object(metadata, "DATA", self.root)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_raw_bytes_are_preserved_and_non_json_is_rejected(self):
        raw = b'{ "extra": 7, "count": 1 }\n'
        path = self.root/"raw.json"
        payload, record = metadata.save_response(response(body=raw), path)
        self.assertEqual(payload["extra"], 7)
        self.assertEqual(path.read_bytes(), raw)
        self.assertNotIn("headers", record)
        for reply in (response(kind="text/html", body=b"CERN login"), response(status=401)):
            with self.assertRaises((ValueError, requests.HTTPError)):
                metadata.save_response(reply, self.root/"invalid.json")
        self.assertFalse((self.root/"invalid.json").exists())

    def test_dcs_csv_roundtrip_validation_and_hash(self):
        raw = b'{ "1": [[1, 4], [7, 7]], "2": [] }\n'
        session = Mock(); session.get.return_value = response(body=raw)
        with redirect_stdout(StringIO()):
            metadata.dcs_collect([2025], session=session)
            metadata.dcs_collect([2025], offline=True)
        self.assertEqual((self.root/"dcs/raw/2025.json").read_bytes(), raw)
        table = pd.read_csv(self.root/"dcs/summary/2025.csv")
        self.assertEqual(table.values.tolist(), [[1, 1, 4], [1, 7, 7]])
        (self.root/"dcs/raw/2025.json").write_text('{"1": [[1, 9]]}')
        with self.assertRaisesRegex(ValueError, "changed"):
            metadata.dcs_collect([2025], offline=True)
        for pair in ([2, 1], [0, 3], [1.2, 3], [True, 3]):
            session.get.return_value = response({"1": [pair]})
            with self.assertRaisesRegex(ValueError, "inclusive"):
                metadata.dcs_collect([2024], session=session)

    def test_oms_pagination_keeps_filters_and_uses_canonical_host(self):
        session = Mock()
        session.get.side_effect = [
            oms_page([dict(run_number=7, lumisection_number=1)], total=2,
                     next_link="https://internal.invalid/lumisections?page[offset]=1&page[limit]=1&filter[run_number][EQ]=999"),
            oms_page([dict(run_number=7, lumisection_number=2)], total=2),
        ]
        rows, records = metadata.oms_pages(session, "lumisections", {"filter[run_number][EQ]": 7},
                                           self.root/"raw", "lumisections")
        self.assertEqual(len(rows), 2); self.assertEqual(len(records), 2)
        self.assertTrue(all(call.args[0] == metadata.OMS+"lumisections" for call in session.get.call_args_list))
        self.assertEqual(session.get.call_args_list[1].kwargs["params"]["filter[run_number][EQ]"], 7)

    def test_oms_multifill_nulls_csv_and_reduced_fields(self):
        session = Mock()
        rows = [dict(run_number=7, lumisection_number=i+1, fill_number=fill,
                     cms_active=True, rpc_ready=ready, pileup=10+i,
                     recorded_lumi_per_lumisection=.5, bpix_ready=True)
                for i, (fill, ready) in enumerate([(10, True), (11, None), (None, False)])]
        session.get.side_effect = [oms_page([dict(run_number=7, fill_number=10)]), oms_page(rows),
                                   oms_page([dict(fill_number=10, bunches_colliding=100)]),
                                   oms_page([dict(fill_number=11, bunches_colliding=200)])]
        table = metadata.oms_collect_run(7, session)
        self.assertEqual(table.bunches_colliding.iloc[:2].tolist(), [100, 200])
        self.assertTrue(pd.isna(table.bunches_colliding.iloc[2]))
        self.assertTrue(pd.isna(table.rpc_ready.iloc[1]))
        self.assertNotIn("bpix_ready", table)
        self.assertNotIn("online_cmssw_version", table)
        pd.testing.assert_frame_equal(metadata.oms_summary(7), table)
        self.assertEqual(len(pd.read_csv(self.root/"oms/summary/7.csv")), 3)
        raw = self.root/"oms/raw/7/lumisections_0001.json"
        self.assertIn("bpix_ready", raw.read_text())
        raw.write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            metadata.oms_summary(7)

    def test_oms_refuses_incomplete_pages_and_duplicate_ls(self):
        session = Mock(); session.get.return_value = oms_page([], total=1)
        with self.assertRaisesRegex(ValueError, "pagination"):
            metadata.oms_pages(session, "lumisections", {}, self.root/"raw", "lumisections")
        session.get.side_effect = [oms_page([dict(run_number=7)]),
                                   oms_page([dict(run_number=7, lumisection_number=1)]*2)]
        with self.assertRaisesRegex(ValueError, "identifiers"):
            metadata.oms_collect_run(7, session)

    def test_rr_pages_fetch_only_promptreco_and_preserve_unknown(self):
        session = Mock()
        raw = b'{ "muon-muon": [{"start": 1,"end": 3,"status": "NOTSET"}], "dt-dt": [] }\n'
        session.post.side_effect = [
            response(dict(count=2, datasets=[dict(name="online")])),
            response(dict(count=2, datasets=[dict(name="/PromptReco/Collisions2025/DQM")])),
            response(body=raw),
        ]
        table = metadata.rr_collect_run(7, session)
        self.assertEqual(session.post.call_count, 3)
        self.assertEqual(table.status.tolist(), ["NOTSET"])
        self.assertEqual(table.label.tolist(), ["unknown"])
        self.assertEqual(table[["start", "end"]].values.tolist(), [[1, 3]])
        self.assertEqual((self.root/"rr/raw/7/components_0001.json").read_bytes(), raw)
        self.assertTrue(all(not call.kwargs["allow_redirects"] for call in session.post.call_args_list))

    def test_rr_failed_refetch_and_changed_raw_do_not_reuse_old_manifest(self):
        session = Mock()
        session.post.side_effect = [response(dict(count=1, datasets=[dict(name="/PromptReco/Collisions2025/DQM")])),
                                    response({"muon-muon": [dict(start=1, end=2, status="GOOD")]})]
        self.assertEqual(metadata.rr_collect_run(7, session).label.tolist(), ["good"])
        raw = self.root/"rr/raw/7/components_0001.json"
        raw.write_text("{}")
        with self.assertRaisesRegex(ValueError, "changed"):
            metadata.rr_summary(7)
        session.post.side_effect = [response(dict(count=2, datasets=[dict(name="online")])),
                                    response(dict(count=2, datasets=[]))]
        with self.assertRaisesRegex(ValueError, "pagination"):
            metadata.rr_collect_run(7, session)
        with self.assertRaises(FileNotFoundError):
            metadata.rr_summary(7)

    def test_rr_year_counts_define_good_bad_and_unknown_without_inventing_labels(self):
        def row(run, counts, **extra):
            return dict(run_number=run, name="/PromptReco/Collisions2025/DQM",
                        DatasetTripletCache={"triplet_summary": {"muon-muon": counts}}, **extra)
        rows = [row(1, {"GOOD": 3}), row(2, {"GOOD": 2, "BAD": 1}),
                row(3, {"GOOD": 2, "NOTSET": 1}), row(4, {}), row(5, {"GOOD": 3}, deleted=True)]
        session = Mock(); session.post.return_value = response(dict(count=len(rows), datasets=rows))
        table = metadata.rr_collect_year(2025, session)
        self.assertEqual(table.label.tolist(), ["good", "bad", "unknown", "unknown", "unknown"])
        self.assertEqual(session.post.call_args.kwargs["json"]["filter"],
                         {"name": "/PromptReco/Collisions2025/DQM"})
        pd.testing.assert_frame_equal(metadata.rr_year_summary(2025), table)
        self.assertTrue((self.root/"rr/summary/runs.csv").exists())

    def test_offline_cli_never_authenticates_and_default_year_is_2025(self):
        session = Mock(); session.post.return_value = response(dict(count=0, datasets=[]))
        metadata.rr_collect_year(2025, session)
        with patch.object(sys, "argv", ["fetch_metadata.py", "rr", "--offline"]), \
                patch.object(metadata, "rr_session", side_effect=AssertionError("No auth expected")), \
                redirect_stdout(StringIO()):
            metadata.main()

    def test_partial_run_failures_are_reported_without_losing_success(self):
        with patch.object(sys, "argv", ["fetch_metadata.py", "oms", "--runs", "1", "2", "--offline"]), \
                patch.object(metadata, "oms_summary", side_effect=[ValueError("missing"), pd.DataFrame()]), \
                redirect_stdout(StringIO()):
            metadata.main()
        failures = json.loads((self.root/"oms/summary/failures.json").read_text())
        self.assertEqual(failures, [dict(run_number=1, error="missing")])
        with patch.object(sys, "argv", ["fetch_metadata.py", "oms", "--runs", "1", "--offline"]), \
                patch.object(metadata, "oms_summary", return_value=pd.DataFrame()), \
                redirect_stdout(StringIO()):
            metadata.main()
        self.assertFalse((self.root/"oms/summary/failures.json").exists())

    def test_run_selection_starts_from_rr_before_any_dqm_cache(self):
        folder = self.root/"rr/summary"; folder.mkdir(parents=True)
        pd.DataFrame({"run_number": [7, 8, 9],
                      "rr_dataset": ["/PromptReco/Collisions2025/DQM"]*2 +
                                    ["/PromptReco/Collisions2026/DQM"]}).to_csv(folder/"runs.csv", index=False)
        manifest = self.root/"dataset/runs.csv"
        with patch.object(metadata, "MANIFEST", manifest):
            self.assertEqual(metadata.selected_runs("oms", False), [7, 8, 9])
            self.assertEqual(metadata.selected_runs("oms", False, [2025]), [7, 8])
            manifest.parent.mkdir()
            pd.DataFrame({"run_number": [8, 9, 10],
                          "dataset": ["/Muon0/Run2025F-PromptReco-v1/DQMIO",
                                      "/ZeroBias/Run2026A-PromptReco-v1/DQMIO",
                                      "/Muon0/Run2025G-PromptReco-v1/DQMIO"]}).to_csv(manifest, index=False)
            self.assertEqual(metadata.selected_runs("oms", False), [8])
            (folder/"runs.csv").unlink()
            self.assertEqual(metadata.selected_runs("oms", False), [8, 10])
            manifest.unlink()
            with self.assertRaisesRegex(ValueError, "RR metadata first"):
                metadata.selected_runs("oms", False)


if __name__ == "__main__":
    unittest.main()
