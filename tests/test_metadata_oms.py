"""OMS response preservation, complete pagination and nullable field extraction."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import fetch_oms


def response(payload, *, status=200, content_type="application/vnd.api+json"):
    result = requests.Response()
    result.status_code = status
    result.headers["content-type"] = content_type
    result.url = fetch_oms.API + "lumisections?filter[run_number][EQ]=1"
    # Deliberate whitespace verifies that raw is not parsed and reserialized.
    result._content = (json.dumps(payload, indent=3) + "\n\n").encode()
    return result


def payload(rows, total=None, next_link=None):
    return {"data": [{"attributes": row, "meta": {"row": {
        "recorded_lumi_per_lumisection": {"units": "pb^{-1}"}}}} for row in rows],
        "meta": {"totalResourceCount": len(rows) if total is None else total},
        "links": {"next": next_link}}


def ls(number, **extra):
    return {"run_number": 1, "lumisection_number": number, "fill_number": 10,
            "recorded_lumi_per_lumisection": .25, "pileup": 20., "cms_active": True,
            "beams_stable": True, "dt0_ready": True, "gem_ready": None,
            "unselected_field": "retained only in raw", **extra}


class Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)


class OMSTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/"oms"

    def leading(self, fill=10):
        return [response(payload([{"run_number": 1, "fill_number": fill,
                                   "era": "Run2025C", "cmssw_version": "ONLINE",
                                   "hlt_key": "/example"}]))]

    def filling(self, fill=10, bunches=2448):
        return response(payload([{"fill_number": fill, "bunches_colliding": bunches}]))

    def test_complete_pages_raw_bytes_safe_host_and_processed_fields(self):
        next_link = "https://https://internal.cern.ch/agg/api/v1/lumisections?page[offset]=2&page[limit]=2&filter[run_number][EQ]=999"
        responses = self.leading() + [response(payload([ls(3), ls(1)], 3, next_link)),
                                      response(payload([ls(2, dt0_ready=None)], 3)), self.filling()]
        originals = [item.content for item in responses]
        session = Session(responses)
        table = fetch_oms.fetch_run(1, session, self.root)
        self.assertEqual(table.ls_number.tolist(), [1, 2, 3])
        self.assertEqual(table.bunches_colliding.tolist(), [2448]*3)
        self.assertEqual(table.online_cmssw_version.tolist(), ["ONLINE"]*3)
        self.assertNotIn("cmssw_version", table)
        self.assertNotIn("unselected_field", table)
        self.assertTrue(table.gem_ready.isna().all())
        self.assertTrue(pd.isna(table.dt0_ready.iloc[1]))
        self.assertEqual(str(table.dt0_ready.dtype), "boolean")
        self.assertEqual(table.recorded_lumi_per_lumisection.tolist(), [.25]*3)
        raw = self.root/"raw/1"
        for name, body in zip(("runs.json", "lumisections_0001.json", "lumisections_0002.json", "fills_10.json"), originals):
            self.assertEqual((raw/name).read_bytes(), body)
        source = json.loads((raw/"source.json").read_text())
        self.assertTrue(source["complete"])
        self.assertEqual(len(source["requests"]), 4)
        self.assertEqual(source["units"]["recorded_lumi_per_lumisection"], "pb^{-1}")
        self.assertEqual(session.calls[2][0], fetch_oms.API+"lumisections")
        self.assertEqual(session.calls[2][1]["params"]["filter[run_number][EQ]"], 1)
        self.assertEqual(session.calls[2][1]["params"]["page[offset]"], "2")
        pd.testing.assert_frame_equal(fetch_oms.process_run(1, self.root), table)
        pd.testing.assert_frame_equal(pd.read_parquet(self.root/"processed/1.parquet"), table)

    def test_incomplete_pages_do_not_claim_a_complete_download(self):
        session = Session(self.leading() + [response(payload([ls(1)], 3))])
        with self.assertRaisesRegex(ValueError, "Incomplete OMS pagination"):
            fetch_oms.fetch_run(1, session, self.root)
        self.assertFalse((self.root/"raw/1/source.json").exists())
        self.assertFalse((self.root/"processed/1.parquet").exists())

    def test_pagination_loop_is_rejected(self):
        next_link = "https://invalid/lumisections?page[offset]=0"
        repeated = response(payload([ls(1)], 2, next_link))
        session = Session(self.leading() + [repeated, repeated])
        with self.assertRaisesRegex(ValueError, "did not advance"):
            fetch_oms.fetch_run(1, session, self.root)

    def test_duplicate_or_wrong_run_ls_are_rejected(self):
        for rows in ([ls(1), ls(1)], [ls(1, run_number=2)]):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(ValueError, "LS identifiers"):
                    fetch_oms.fetch_run(1, Session(self.leading()+[response(payload(rows)), self.filling()]), self.root)

    def test_absent_fill_remains_missing_and_empty_run_is_valid(self):
        session = Session(self.leading(None) + [response(payload([ls(1, fill_number=None)]))])
        table = fetch_oms.fetch_run(1, session, self.root)
        self.assertTrue(table.bunches_colliding.isna().all())
        self.assertFalse(list((self.root/"raw/1").glob("fills_*.json")))
        empty = fetch_oms.fetch_run(1, Session(self.leading()+[response(payload([])), self.filling()]), self.root)
        self.assertEqual(len(empty), 0)
        self.assertIn("ls_number", empty)
        self.assertFalse((self.root/"raw/1/lumisections_0002.json").exists())

    def test_http_and_login_html_are_rejected(self):
        for item, exception in [(response({}, status=401), requests.HTTPError),
                                (response({}, content_type="text/html"), ValueError)]:
            with self.subTest(exception=exception):
                with self.assertRaises(exception):
                    fetch_oms.fetch_run(1, Session([item]), self.root)
                self.assertFalse((self.root/"raw/1/runs.json").exists())

    def test_offline_uses_the_manifest_and_rejects_modified_raw(self):
        table = fetch_oms.fetch_run(1, Session(self.leading()+[response(payload([ls(1)])), self.filling()]), self.root)
        folder = self.root/"raw/1"
        (folder/"lumisections_9999.json").write_text(json.dumps(payload([ls(999)])))
        pd.testing.assert_frame_equal(fetch_oms.process_run(1, self.root), table)
        path = folder/"lumisections_0001.json"
        original = path.read_bytes()
        path.write_bytes(original+b" ")
        with self.assertRaisesRegex(ValueError, "raw response changed"):
            fetch_oms.process_run(1, self.root)
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            fetch_oms.process_run(1, self.root)

    def test_ls_fill_metadata_can_differ_from_run_header(self):
        rows = [ls(1), ls(2, fill_number=11), ls(3, fill_number=None)]
        session = Session(self.leading()+[response(payload(rows)), self.filling(), self.filling(11, 1200)])
        table = fetch_oms.fetch_run(1, session, self.root)
        self.assertEqual(table.run_fill_number.tolist(), [10]*3)
        self.assertEqual(table.fill_number.iloc[:2].tolist(), [10, 11])
        self.assertTrue(pd.isna(table.fill_number.iloc[2]))
        self.assertEqual(table.bunches_colliding.iloc[:2].tolist(), [2448, 1200])
        self.assertTrue(pd.isna(table.bunches_colliding.iloc[2]))
        self.assertEqual([call[1]["params"]["filter[fill_number][EQ]"] for call in session.calls[2:]], [10, 11])
        pd.testing.assert_frame_equal(fetch_oms.process_run(1, self.root), table)


if __name__ == "__main__":
    unittest.main()
