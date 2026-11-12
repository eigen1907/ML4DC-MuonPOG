"""Small annual-metadata tests; no CERN authentication or downloads."""

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import fetch_metadata as metadata


def response(value=None, *, body=None, status=200, kind="application/json"):
    result = requests.Response()
    result.status_code, result.url = status, metadata.OMS+"test"
    result.headers["Content-Type"] = kind
    result._content = body if body is not None else json.dumps(value).encode()
    return result


def page(rows, *, total=None, next_link=None):
    return response({"data": [{"attributes": row} for row in rows],
                     "meta": {"totalResourceCount": len(rows) if total is None else total},
                     "links": {"next": next_link}})


def ls(run, number, fill, year=2025, **extra):
    return dict(run_number=run, lumisection_number=number, fill_number=fill,
                start_time=f"{year}-06-01T00:00:00Z", **extra)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = patch.object(metadata, "DATA", self.root)
        patcher.start(); self.addCleanup(patcher.stop)

    def test_json_response_rejects_failed_auth_and_html(self):
        self.assertEqual(metadata.json_response(response(body=b'{ "extra": 7 }\n')), {"extra": 7})
        for reply in (response(kind="text/html", body=b"CERN login"), response(status=401)):
            with self.assertRaises((ValueError, requests.HTTPError)):
                metadata.json_response(reply)

    def test_dcs_preserves_original_bytes_without_sidecars(self):
        raw = b'{ "1": [[1, 4], [7, 7]], "2": [] }\n'
        session = Mock(); session.get.return_value = response(body=raw)
        metadata.dcs_collect([2025], session=session)
        self.assertEqual(metadata.metadata_path("dcs", 2025).read_bytes(), raw)
        self.assertEqual([path.name for path in (self.root/"dcs").iterdir()], ["2025.json"])
        for bounds in ([2, 1], [0, 3], [1.2, 3], [True, 3]):
            session.get.return_value = response({"1": [bounds]})
            with self.assertRaisesRegex(ValueError, "inclusive"):
                metadata.dcs_collect([2025], session=session)
            self.assertEqual(metadata.metadata_path("dcs", 2025).read_bytes(), raw)

    def test_oms_pages_keep_filter_and_public_host(self):
        session = Mock(); session.get.side_effect = [
            page([ls(7, 1, 10)], total=2,
                 next_link="https://internal.invalid/ls?page[offset]=1&page[limit]=1&filter[run_number][EQ]=999"),
            page([ls(7, 2, 10)], total=2),
        ]
        rows = metadata.oms_pages(session, "lumisections", {"filter[run_number][EQ]": 7})
        self.assertEqual([row["lumisection_number"] for row in rows], [1, 2])
        self.assertTrue(all(set(row) == set(metadata.OMS_FIELDS) for row in rows))
        self.assertTrue(all(call.args[0] == metadata.OMS+"lumisections" for call in session.get.call_args_list))
        self.assertEqual(session.get.call_args_list[1].kwargs["params"]["filter[run_number][EQ]"], 7)
        session.get.side_effect = None; session.get.return_value = page([], total=1)
        with self.assertRaisesRegex(ValueError, "pagination"):
            metadata.oms_pages(session, "lumisections", {})

    def test_oms_saves_only_ls_fields_and_replaces_one_run(self):
        session = Mock(); session.get.side_effect = [
            page([ls(7, 1, 10, cms_active=True, rpc_ready=True, bpix_ready=True),
                  ls(7, 2, 11, cms_active=None), ls(7, 3, None)]),
        ]
        value = metadata.oms_collect_run(7, session)
        self.assertIsNone(value["lumisections"][1]["cms_active"])
        self.assertEqual(session.get.call_count, 1)
        self.assertEqual(session.get.call_args.args[0], metadata.OMS+"lumisections")
        self.assertEqual(session.get.call_args_list[0].kwargs["params"]["fields"], ",".join(metadata.OMS_FIELDS))
        raw = json.loads(metadata.metadata_path("oms", 2025).read_text())
        self.assertEqual(set(raw), {"lumisections"})
        self.assertEqual(set(raw["lumisections"][0]), set(metadata.OMS_FIELDS))
        self.assertNotIn("attributes", raw["lumisections"][0])
        session.get.reset_mock(); session.get.side_effect = [page([ls(8, 1, 10)])]
        value = metadata.oms_collect_run(8, session)
        self.assertEqual(session.get.call_count, 1)
        self.assertEqual([row["run_number"] for row in value["lumisections"]], [7, 7, 7, 8])
        session.get.side_effect = [page([ls(7, 4, 11)])]
        value = metadata.oms_collect_run(7, session)
        self.assertEqual([(row["run_number"], row["lumisection_number"])
                          for row in value["lumisections"]], [(7, 4), (8, 1)])
        self.assertEqual(json.loads(metadata.metadata_path("oms", 2025).read_text()), value)
        self.assertEqual([p.name for p in (self.root/"oms").iterdir()], ["2025.json"])

    def test_oms_bad_identifiers_or_cross_year_do_not_change_saved_file(self):
        session = Mock(); session.get.side_effect = [page([ls(7, 1, None)])]
        metadata.oms_collect_run(7, session)
        before = metadata.metadata_path("oms", 2025).read_bytes()
        for rows, message in [([ls(7, 1, None)]*2, "identifiers"),
                              ([ls(7, 1, None), ls(7, 2, None, 2026)], "multiple years"),
                              ([dict(run_number=7, lumisection_number=1)], "timestamps")]:
            session.get.side_effect = [page(rows)]
            with self.assertRaisesRegex(ValueError, message):
                metadata.oms_collect_run(7, session)
            self.assertEqual(metadata.metadata_path("oms", 2025).read_bytes(), before)

    def test_oms_batch_queries_only_ls_and_writes_annual_files_once(self):
        session = Mock(); session.get.side_effect = [page([ls(7, 1, 10)]), page([ls(8, 1, 10)])]
        with patch.object(metadata, "write_json", wraps=metadata.write_json) as write:
            self.assertEqual(metadata.oms_collect_runs([7, 8], session), (2, 0))
        self.assertEqual(session.get.call_count, 2)
        self.assertTrue(all(call.args[0] == metadata.OMS+"lumisections" for call in session.get.call_args_list))
        self.assertEqual(write.call_count, 1)
        value = json.loads(metadata.metadata_path("oms", 2025).read_text())
        self.assertEqual([row["run_number"] for row in value["lumisections"]], [7, 8])

    def test_oms_failed_run_does_not_mutate_batch_cache(self):
        cache, session = {}, Mock()
        session.get.side_effect = [page([ls(7, 1, None)])]
        metadata.oms_collect_run(7, session, cache=cache)
        previous = json.dumps(cache)
        session.get.side_effect = [response(status=500)]
        with self.assertRaises(requests.HTTPError):
            metadata.oms_collect_run(8, session, cache=cache)
        self.assertEqual(json.dumps(cache), previous)
        self.assertFalse(metadata.metadata_path("oms", 2025).exists())

    def test_rr_run_component_matrix_and_unknown_verdicts(self):
        def row(run, counts, **extra):
            return dict(run_number=run, name="/PromptReco/Collisions2025/DQM",
                        DatasetTripletCache={"triplet_summary": counts}, **extra)
        rows = [row(1, {"muon-muon": {"GOOD": 3}, "dt-dt": {"GOOD": 2, "BAD": 1}}),
                row(2, {"muon-muon": {"GOOD": 2, "NOTSET": 1}}),
                row(3, {"muon-muon": {"BAD": 2}}), row(4, {}),
                row(5, {"muon-muon": {"GOOD": 3}}, deleted=True)]
        rows[0]["Run"] = {"unneeded": "large embedded OMS object"}
        rows[0]["dataset_attributes"] = {"muon_state": "COMPLETED"}
        counts = rows[0]["DatasetTripletCache"]["triplet_summary"]
        counts["muon-muon"].update(comments=["not stored"], causes=["not stored"])
        counts["unused-component"] = {"GOOD": 3}
        session = Mock(); session.post.return_value = response(dict(count=5, datasets=rows))
        value = metadata.rr_collect_year(2025, session)
        for component, expected in [("muon", ["good", "unknown", "bad", "unknown", "unknown"]),
                                    ("dt", ["bad", "unknown", "unknown", "unknown", "unknown"])]:
            self.assertEqual(["unknown" if row["deleted"] else metadata.rr_verdict(row["components"][component])
                              for row in value["runs"]], expected)
        self.assertEqual(session.post.call_count, 1)
        self.assertEqual(session.post.call_args.kwargs["json"]["filter"], {"name": "/PromptReco/Collisions2025/DQM"})
        self.assertEqual([p.name for p in (self.root/"rr").iterdir()], ["2025.json"])
        raw = json.loads(metadata.metadata_path("rr", 2025).read_text())
        self.assertEqual(raw, value)
        self.assertEqual(set(raw), {"runs"})
        self.assertEqual(set(raw["runs"][0]), {"run_number", "deleted", "components"})
        self.assertEqual(set(raw["runs"][0]["components"]), set(metadata.RR_COMPONENTS))
        self.assertEqual(raw["runs"][0]["components"]["muon"], {"GOOD": 3})

    def test_rr_pagination_and_failed_refresh_preserve_previous_annual_file(self):
        def row(run):
            return dict(run_number=run, name="/PromptReco/Collisions2025/DQM")
        session = Mock(); session.post.side_effect = [response(dict(count=2, datasets=[row(1)])),
                                                    response(dict(count=2, datasets=[row(2)]))]
        metadata.rr_collect_year(2025, session)
        self.assertEqual([call.kwargs["json"]["page"] for call in session.post.call_args_list], [0, 1])
        before = metadata.metadata_path("rr", 2025).read_bytes()
        session.post.side_effect = [response(dict(count=2, datasets=[row(1)])), response(dict(count=2, datasets=[]))]
        with self.assertRaisesRegex(ValueError, "pagination"):
            metadata.rr_collect_year(2025, session)
        self.assertEqual(metadata.metadata_path("rr", 2025).read_bytes(), before)

    def test_cli_rejects_removed_offline_option_and_rr_run_selection(self):
        for arguments in (["rr", "--offline"], ["rr", "--runs", "7"]):
            with patch.object(sys, "argv", ["fetch_metadata.py", *arguments]), \
                    patch.object(metadata, "rr_session", side_effect=AssertionError("No auth expected")), \
                    redirect_stderr(StringIO()), self.assertRaises(SystemExit):
                metadata.main()

    def test_selected_runs_uses_annual_rr_and_optional_muon0_cache(self):
        metadata.write_json(metadata.metadata_path("rr", 2025), {"runs": [{"run_number": 7}, {"run_number": 8}]})
        metadata.write_json(metadata.metadata_path("rr", 2026), {"runs": [{"run_number": 9}]})
        manifest = self.root/"dataset/runs.csv"
        with patch.object(metadata, "MANIFEST", manifest):
            self.assertEqual(metadata.selected_runs(), [7, 8, 9])
            self.assertEqual(metadata.selected_runs([2025]), [7, 8])
            manifest.parent.mkdir()
            manifest.write_text("run_number,dataset\n8,/Muon0/Run2025F-PromptReco-v1/DQMIO\n"
                                "9,/ZeroBias/Run2026A-PromptReco-v1/DQMIO\n10,/Muon0/Run2025G-PromptReco-v1/DQMIO\n")
            self.assertEqual(metadata.selected_runs(), [8])
            self.assertEqual(metadata.selected_runs([2026]), [9])
            for path in (self.root/"rr").glob("*.json"): path.unlink()
            self.assertEqual(metadata.selected_runs(), [8, 10])
            manifest.unlink()
            with self.assertRaisesRegex(ValueError, "RR metadata first"):
                metadata.selected_runs()


if __name__ == "__main__":
    unittest.main()
