"""RR pagination and raw/selected separation, without CERN authentication."""

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
import fetch_rr


def response(value, *, body=None):
    reply = requests.Response()
    reply.status_code = 200
    reply.url = fetch_rr.API + "/test"
    reply.headers["Content-Type"] = "application/json; charset=utf-8"
    reply._content = body if body is not None else json.dumps(value).encode()
    return reply


class RunRegistryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)/"rr"
        patcher = patch.object(fetch_rr, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_all_pages_all_datasets_and_exact_raw_bytes(self):
        names = ["online", "/PromptReco/Collisions2025/DQM"]
        raw = b'{ "muon-muon": [{"start": 1, "end": 2, "status": "NOTSET", "comment": null, "cause": ""}], "dt-dt": [] }\n'
        session = Mock()
        session.post.side_effect = [
            response({"count": 2, "pages": 2, "datasets": [{"name": names[0]}]}),
            response({"count": 2, "pages": 2, "datasets": [{"name": names[1]}]}),
            response(None, body=raw),
            response({"muon-muon": [dict(start=1, end=10, status="BAD",
                                          comment="Barrel issue", cause="Reconstruction")],
                      "csc-csc": [dict(start=1, end=10, status="GOOD")]}),
        ]
        fetch_rr.collect_run(123, session)
        frame = fetch_rr.process_run(123)
        self.assertEqual((self.root/"raw/123/components_0001.json").read_bytes(), raw)
        self.assertEqual(session.post.call_count, 4)
        self.assertEqual([call.kwargs["json"]["page"] for call in session.post.call_args_list[:2]], [0, 1])
        self.assertTrue(all(not call.kwargs["allow_redirects"] for call in session.post.call_args_list))
        self.assertEqual([call.kwargs["json"]["dataset_name"] for call in session.post.call_args_list[2:]], names)
        source = json.loads((self.root/"raw/123/source.json").read_text())
        self.assertEqual(len(source["requests"]), 4)
        self.assertEqual(frame.rr_dataset.tolist(), names)
        self.assertEqual(frame.status.tolist(), ["NOTSET", "BAD"])
        self.assertTrue(pd.isna(frame.comment.iloc[0]))
        self.assertEqual(frame.comment.iloc[1], "Barrel issue")
        self.assertEqual(frame[["start", "end"]].values.tolist(), [[1, 2], [1, 10]])
        self.assertEqual(len(frame), 2)  # Intervals remain intervals, not expanded LS.
        pd.testing.assert_frame_equal(frame, pd.read_parquet(self.root/"processed/123.parquet"))

    def test_offline_processing_never_authenticates(self):
        session = Mock()
        session.post.side_effect = [response({"count": 1, "pages": 1, "datasets": [{"name": "online"}]}),
                                    response({"muon-muon": [dict(start=5, end=5, status="NO VALUE FOUND")]})]
        fetch_rr.collect_run(123, session)
        with patch.object(sys, "argv", ["fetch_rr.py", "--runs", "123", "--offline"]), \
                patch.object(fetch_rr, "authenticated_session", side_effect=AssertionError("No login expected")), \
                redirect_stdout(StringIO()):
            fetch_rr.main()
        frame = pd.read_parquet(self.root/"processed/123.parquet")
        self.assertEqual(frame.status.tolist(), ["NO VALUE FOUND"])

    def test_empty_catalogue_and_missing_muon_component_stay_empty(self):
        for run, replies in [
            (123, [response({"count": 0, "pages": 0, "datasets": []})]),
            (124, [response({"count": 1, "pages": 1, "datasets": [{"name": "online"}]}),
                   response({"cms-cms": [dict(start=1, end=10, status="GOOD")]})]),
        ]:
            with self.subTest(run=run):
                session = Mock(); session.post.side_effect = replies
                fetch_rr.collect_run(run, session)
                frame = fetch_rr.process_run(run)
                self.assertTrue(frame.empty)
                self.assertEqual(list(frame), fetch_rr.COLUMNS)
                self.assertEqual(str(frame.run_number.dtype), "int64")

    def test_incomplete_pagination_is_rejected(self):
        session = Mock()
        session.post.side_effect = [response({"count": 2, "datasets": [{"name": "online"}]}),
                                    response({"count": 2, "datasets": []})]
        with self.assertRaisesRegex(ValueError, "pagination ended early"):
            fetch_rr.collect_run(123, session)
        self.assertFalse((self.root/"raw/123/source.json").exists())

    def test_invalid_interval_is_rejected_without_inventing_a_label(self):
        session = Mock()
        session.post.side_effect = [response({"count": 1, "datasets": [{"name": "online"}]}),
                                    response({"muon-muon": [dict(start=10, end=1, status="GOOD")]})]
        fetch_rr.collect_run(123, session)
        with self.assertRaisesRegex(ValueError, "Invalid Run Registry Muon interval"):
            fetch_rr.process_run(123)

    def test_failed_refetch_invalidates_the_previous_manifest(self):
        session = Mock()
        session.post.side_effect = [response({"count": 1, "datasets": [{"name": "online"}]}),
                                    response({"muon-muon": [dict(start=1, end=2, status="GOOD")]})]
        fetch_rr.collect_run(123, session)
        fetch_rr.process_run(123)
        session.post.side_effect = [response({"count": 2, "datasets": [{"name": "online"}]}),
                                    response({"count": 2, "datasets": []})]
        with self.assertRaisesRegex(ValueError, "pagination ended early"):
            fetch_rr.collect_run(123, session)
        with self.assertRaises(FileNotFoundError):
            fetch_rr.process_run(123)

    def test_offline_processing_rejects_changed_response_bytes(self):
        session = Mock()
        session.post.side_effect = [response({"count": 1, "datasets": [{"name": "online"}]}),
                                    response({"muon-muon": [dict(start=1, end=2, status="GOOD")]})]
        fetch_rr.collect_run(123, session)
        path = self.root/"raw/123/components_0001.json"
        path.write_bytes(path.read_bytes() + b"\n")  # Still JSON, different original bytes.
        with self.assertRaisesRegex(ValueError, "differs from its provenance"):
            fetch_rr.process_run(123)


if __name__ == "__main__":
    unittest.main()
