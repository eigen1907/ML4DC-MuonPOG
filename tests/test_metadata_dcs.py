"""DCS raw-byte preservation and inclusive interval extraction."""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_dcs


class DCSTests(unittest.TestCase):
    def test_extracts_ranges_without_expansion_and_rejects_invalid_input(self):
        actual = fetch_dcs.intervals({"12": [[2, 5], [9, 9]], "13": []})
        self.assertEqual(actual.to_dict("records"), [
            {"run_number": 12, "first_ls": 2, "last_ls": 5},
            {"run_number": 12, "first_ls": 9, "last_ls": 9}])
        self.assertEqual(actual.dtypes.tolist(), ["int64"] * 3)
        for bad in ({}, [], {"x": [[1, 2]]}, {"0": [[1, 2]]}, {"1": [1]},
                    {"1": [[2, 1]]}, {"1": [[0, 2]]}, {"1": [[True, 2]]},
                    {"1": [[1., 2]]}, {"1": [[1, 2, 3]]}):
            with self.subTest(payload=bad), self.assertRaises(ValueError):
                fetch_dcs.intervals(bad)

    def test_online_preserves_response_bytes_and_offline_needs_no_network(self):
        body = b'{  "12" : [[2, 5], [9,9]] }\n'
        response = requests.Response()
        response.status_code = 200
        response.url = fetch_dcs.source_url(2025)
        response.headers["Content-Type"] = "application/json"
        response._content = body
        session = Mock()
        session.get.return_value = response
        with tempfile.TemporaryDirectory() as folder, patch.object(fetch_dcs, "ROOT", Path(folder)):
            fetch_dcs.collect([2025], session=session)
            raw = Path(folder) / "raw/2025.json"
            self.assertEqual(raw.read_bytes(), body)
            source = json.loads(raw.with_suffix(".source.json").read_text())
            self.assertEqual(source["year"], 2025)
            self.assertEqual(source["sha256"], hashlib.sha256(body).hexdigest())
            self.assertIn("not Muon", source["semantics"])
            output = Path(folder) / "processed/2025.parquet"
            original = pd.read_parquet(output)
            output.unlink()
            session.reset_mock()
            fetch_dcs.collect([2025], offline=True, session=session)
            session.get.assert_not_called()
            pd.testing.assert_frame_equal(original, pd.read_parquet(output))
            self.assertEqual(raw.read_bytes(), body)
            raw.write_bytes(b'{"12": [[2, 6]]}')
            with self.assertRaisesRegex(ValueError, "differs from its provenance"):
                fetch_dcs.collect([2025], offline=True)
            raw.write_bytes(body)
            raw.with_suffix(".source.json").unlink()
            with self.assertRaises(FileNotFoundError):
                fetch_dcs.collect([2025], offline=True)

    def test_url_uses_actual_2023_filename_and_rejects_unknown_year(self):
        self.assertIn("eraBCD_366403_370790_DCSOnly_TkPx", fetch_dcs.source_url(2023))
        self.assertIn("Collisions26_13p6TeV_Latest", fetch_dcs.source_url(2026))
        with self.assertRaises(ValueError):
            fetch_dcs.source_url(2022)


if __name__ == "__main__":
    unittest.main()
