"""Small fixtures for the cached-data, historical-moment and run-split contract."""

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import data


class DataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for key, value in {"RAW": self.root / "raw", "MANIFEST": self.root / "runs.csv",
                           "RUN_LABELS": self.root / "labels.csv", "DCS_PATH": self.root / "dcs.json"}.items():
            patcher = patch.object(data, key, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def cache(self):
        self.names = [f"{stem}.parquet" for stem in data.MODEL_MES] + [
            "Muons__MuonRecoOneHLT__GlbMuon_Glb_eta.parquet",
            "Muons__MuonRecoOneHLT__GlbMuon_Glb_pt.parquet",
        ]
        rows, labels = [], []
        for era, offset in (("B", 100), ("C", 200)):
            dataset = f"/Muon0/Run2025{era}-PromptReco-v1/DQMIO"
            for index in range(7):
                run = offset + index
                label = "good" if index < 6 else "bad_well_known" if era == "B" else "bad_ambiguous"
                rows.append({"dataset": dataset, "run_number": run, "ls_count": 2, "ls_completeness": 1.})
                labels.append({"run_number": run, "label": label})
                for name in self.names:
                    self.write_me(dataset, run, name)
        manifest = pd.DataFrame(rows)
        manifest.to_csv(data.MANIFEST, index=False)
        pd.DataFrame(labels).to_csv(data.RUN_LABELS, index=False)
        data.DCS_PATH.write_text(json.dumps({str(row["run_number"]): [[1, 2]] for row in rows}))
        return manifest

    def write_me(self, dataset, run, name, *, rows=None, entries=None, ls=None, low=0., high=2., bins=2):
        folder = data.RAW / dataset.strip("/").replace("/", "__") / str(run)
        folder.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"ls_number": [2, 1] if ls is None else ls,
                      "entries": [0, 6001] if entries is None else entries,
                      "data": [[0, 0], [1, 3]] if rows is None else rows,
                      "x_min": low, "x_max": high, "x_bin": bins}).to_parquet(folder / name, index=False)
        return folder / name

    def test_historical_moments_and_zero_information(self):
        moments = data._moments(np.array([[1, 3], [0, 0]], dtype=np.int32), 0, 2)
        np.testing.assert_allclose(moments, [[.625, np.sqrt(.046875)], [0, 0]])
        eta = data._moments(np.array([[1, 0, 0, 0], [0, 2, 2, 0]]), -2, 2, eta=True)
        np.testing.assert_array_equal(eta, [[.75, 0], [.25, 0]])
        # Scalar dtypes must not change the historical float32 bin positions.
        np.testing.assert_array_equal(data._moments(np.array([[1, 2, 3]]), 0, 1),
                                      data._moments(np.array([[1, 2, 3]]), np.float64(0), np.float64(1)))

    def test_single_read_split_before_cache_filter_and_root_entries(self):
        manifest = self.cache()
        missing = data.RAW / manifest.iloc[-2].dataset.strip("/").replace("/", "__") / "205" / self.names[-1]
        missing.unlink()
        good = manifest[~manifest.run_number.isin([106, 206])].copy()
        good["era"] = good.dataset.str.extract(r"Run(2025[A-Z])")[0]
        train, holdout = train_test_split(good, train_size=.7, random_state=data.SEED, stratify=good.era)
        validation, test = train_test_split(holdout, train_size=.5, random_state=data.SEED, stratify=holdout.era)
        expected = {"train": set(train.run_number), "validation": set(validation.run_number),
                    "test": set(test.run_number) | {106, 206}}
        with patch.object(pd, "read_parquet", wraps=pd.read_parquet) as read:
            dataset = data.load_dataset()
        self.assertEqual(read.call_count, 13 * 8)
        self.assertEqual(len(dataset["frame"]), 26)
        self.assertEqual(dataset["features"].shape, (26, 16))
        self.assertEqual(dataset["me_names"], list(data.MODEL_MES))
        for key, members in expected.items():
            self.assertEqual(dataset["run_split"][key], sorted(members - {205}))
        frame = dataset["frame"]
        self.assertTrue(frame.loc[frame.label.ne("good"), "split"].eq("test").all())
        self.assertFalse(set(dataset["run_split"]["train"]) & set(dataset["run_split"]["test"]))
        np.testing.assert_array_equal(frame.ls_number[:2], [1, 2])
        np.testing.assert_array_equal(frame.phi_entries[:2], [6001, 0])
        np.testing.assert_array_equal(frame.dcs_pass[:2], [True, True])
        self.assertTrue(frame.exclusion.eq("Included").all())
        self.assertNotIn("eligible", frame)
        self.assertNotIn("entries_pass", frame)
        np.testing.assert_array_equal(dataset["totals"][:2, 0], [4, 0])
        np.testing.assert_array_equal(dataset["root_entries"][:2], [[6001]*8, [0]*8])
        np.testing.assert_array_equal(dataset["histograms"][0][:2], [[1, 3], [0, 0]])
        self.assertEqual(dataset["histograms"][0].dtype, np.int32)
        np.testing.assert_array_equal(dataset["axes"][0]["edges"], [0, 1, 2])
        np.testing.assert_allclose(dataset["features"][:2, :2], [[.625, np.sqrt(.046875)], [0, 0]])

    def test_feature_dimensions_and_entries_follow_me_names(self):
        manifest = self.cache()
        dataset, run = manifest.iloc[0].dataset, 100
        hm = next(me for me in data.MODEL_MES if me.endswith("__GlbGlbMuon_HM"))
        self.write_me(dataset, run, f"{hm}.parquet", entries=[0, 17], rows=[[0, 0], [4, 0]])
        # Reordering or dropping an ME must not reinterpret eta/phi metadata.
        names = tuple(reversed(data.MODEL_MES[1:]))
        with patch.object(data, "MODEL_MES", names):
            loaded = data.load_dataset()
        self.assertEqual(loaded["features"].shape, (28, 2 * len(names)))
        self.assertEqual(loaded["me_names"], list(names))
        np.testing.assert_array_equal(loaded["frame"].eta_entries[:2], [6001, 0])
        np.testing.assert_array_equal(loaded["frame"].phi_entries[:2], [6001, 0])
        hm_index = names.index(hm)
        np.testing.assert_array_equal(loaded["root_entries"][:2, hm_index], [17, 0])
        np.testing.assert_array_equal(loaded["features"][0, 2*hm_index:2*hm_index+2], [.25, 0])

    def test_missing_first_cache_does_not_define_the_me_schema(self):
        manifest = self.cache()
        first = manifest.iloc[0]
        data.raw_folder(first.dataset, first.run_number, root=data.RAW).joinpath(self.names[0]).unlink()
        loaded = data.load_dataset()
        self.assertEqual(len(loaded['frame']), 26)
        self.assertNotIn(first.run_number, loaded['frame'].run_number.tolist())
        self.assertEqual(loaded['me_names'], list(data.MODEL_MES))

    def test_rejects_invalid_counts_ls_axes_and_shared_entries(self):
        manifest = self.cache()
        dataset, run = manifest.iloc[0].dataset, 100
        name = self.names[0]
        for changes, message in [({"rows": [[0, 0], [1.5, 3]]}, "integer histogram"),
                                 ({"rows": [[0, 0], [-1, 3]]}, "integer histogram"),
                                 ({"ls": [1, 1]}, "duplicate LS"),
                                 ({"ls": [2, 3]}, "coverage differs"),
                                 ({"bins": 3}, "histogram axis"),
                                 ({"entries": [0, 6002]}, "ROOT entries differ")]:
            with self.subTest(changes=changes):
                self.write_me(dataset, run, name, **changes)
                with self.assertRaisesRegex(ValueError, message):
                    data.load_dataset()
                self.write_me(dataset, run, name)
        self.write_me(manifest.iloc[1].dataset, 101, name, high=3.)
        with self.assertRaisesRegex(ValueError, "axis differs between runs"):
            data.load_dataset()

    def test_dcs_selection_keeps_excluded_and_missing_rows(self):
        frame = pd.DataFrame({"run_number": [1, 1, 1, 1, 2], "ls_number": [1, 2, 3, 4, 1]})
        entries = np.array([0, 0, 1, 5000, 6000])
        result = data.select(frame, {"1": [[2, 3]]}, entries)
        np.testing.assert_array_equal(result.dcs_pass, [False, True, True, False, False])
        np.testing.assert_array_equal(result.phi_entries, entries)
        self.assertEqual(set(result), {"phi_entries", "dcs_run_present", "dcs_pass", "exclusion"})
        self.assertEqual(result.exclusion.tolist(), ["DCS LS excluded", "Included", "Included",
                                                   "DCS LS excluded", "DCS run absent"])
        # Counts have no role in DCS selection, even for completely empty LS.
        empty = data.select(frame, {"1": [[2, 3]]}, np.zeros(len(frame)))
        np.testing.assert_array_equal(empty.dcs_pass, result.dcs_pass)
        pd.testing.assert_series_equal(empty.exclusion, result.exclusion)
        data.DCS_PATH.write_text('{"1": [[1, 0]]}')
        with self.assertRaisesRegex(ValueError, "inclusive DCS"):
            data.load_dcs()

    def test_validation_roles_split_before_removing_special_controls(self):
        runs = list(range(390001, 390025))
        runs[5], runs[17] = data.SPECIAL_RUNS
        frame = pd.DataFrame({"run_number": runs, "era": ["2025B"]*12 + ["2025C"]*12,
                              "split": "validation", "label": "good"})
        monitor, calibration = train_test_split(frame, test_size=.5, random_state=data.SEED, stratify=frame.era)
        roles = data.validation_roles(pd.concat([frame, frame], ignore_index=True))
        self.assertEqual(roles["monitor_runs"], sorted(set(monitor.run_number) - set(data.SPECIAL_RUNS)))
        self.assertEqual(roles["calibration_runs"], sorted(set(calibration.run_number) - set(data.SPECIAL_RUNS)))
        self.assertFalse(set(roles["monitor_runs"]) & set(roles["calibration_runs"]))
        self.assertEqual(set(roles), {"monitor_runs", "calibration_runs"})

    def test_fingerprint_includes_exact_counts_labels_and_dcs_bytes(self):
        data.DCS_PATH.write_text('{"1": [[1, 1]]}')
        frame = pd.DataFrame({"dataset": ["/test"], "run_number": [1], "ls_number": [1],
                              "split": ["train"], "label": ["good"]})
        dataset = {"frame": frame, "histograms": [np.array([[1, 2]], dtype=np.int32)]}
        expected = hashlib.sha256(data.DCS_PATH.read_bytes())
        expected.update(frame[data.IDS + ["split", "label"]].to_csv(index=False).encode())
        expected.update(memoryview(dataset["histograms"][0]))
        original = data.fingerprint(dataset)
        self.assertEqual(original, expected.hexdigest())
        dataset["histograms"][0][0, 0] += 1
        self.assertNotEqual(original, data.fingerprint(dataset))
        dataset["histograms"][0][0, 0] -= 1
        frame.loc[0, "label"] = "bad_well_known"
        self.assertNotEqual(original, data.fingerprint(dataset))


if __name__ == "__main__":
    unittest.main()
