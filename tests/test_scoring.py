"""Small leakage, missing-evidence, window and cohort checks."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from data import SPECIAL_RUNS
from models import residuals
from scoring import (base_terms, calibrate, histogram_terms, mean_score,
                     require_alignment, summarize, window_mean)


class FixedPrediction:
    def predict(self, values):
        self.last_input = values.copy()
        return np.float32(.125) * values + np.linspace(-.2, .3, values.shape[1], dtype=np.float32)


class ScoringTests(unittest.TestCase):
    def test_histogram_residuals_reuse_one_decoder_pass_per_batch(self):
        class FixedDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def forward(self, inputs):
                self.calls += 1
                return [{'probabilities': torch.tensor([[[.3, .7]]]).expand(len(inputs), -1, -1),
                         'concentration': torch.tensor(10.)} for _ in range(3)]

        model = FixedDecoder()
        inputs = torch.zeros(5, 1)
        histograms = [np.array([[1, 3], [0, 0], [2, 1], [1, 0], [0, 4]]) for _ in range(3)]
        expected = residuals(histograms, model(inputs))
        model.calls = 0
        actual = histogram_terms(model, inputs, histograms, batch_size=2)
        self.assertEqual(model.calls, 3)
        self.assertFalse(model.training)
        self.assertEqual(set(actual), {'mse', 'weighted'})
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key].numpy())

    def test_empty_shapes_remain_missing_and_available_mes_have_equal_weights(self):
        terms = np.array([[np.nan, 4.], [np.nan, np.nan], [2., 4.]])
        np.testing.assert_allclose(mean_score(terms), [4, np.nan, 3], equal_nan=True)
        for invalid in ([[1., np.inf]], [1., 2.]):
            with self.assertRaises(ValueError):
                mean_score(invalid)

    def test_calibration_uses_supplied_good_rows_and_dcs_missing_flags_stay_unknown(self):
        frame = pd.DataFrame({'label': ['good']*6 + ['bad', 'good'],
                              'split': ['validation']*3 + ['test']*5,
                              'dcs_pass': [True]*5 + [False, True, True]}, index=np.arange(8)*3 + 10)
        terms = np.repeat(np.array([1., 2., 3., 1e6, np.nan, 1e7, 1e8, 2.95])[:, None], 2, axis=1)
        calibration = np.array([True, True, True, False, False, False, False, False])
        original_frame, original_terms = frame.copy(deep=True), terms.copy()
        scores, cut = calibrate(frame, terms, calibration)
        self.assertAlmostEqual(cut, 2.95)
        pd.testing.assert_frame_equal(frame, original_frame)
        np.testing.assert_array_equal(terms, original_terms)
        pd.testing.assert_index_equal(scores.index, frame.index)
        self.assertTrue(scores.flagged.iloc[3])
        self.assertTrue(pd.isna(scores.flagged.iloc[4]))
        self.assertTrue(pd.isna(scores.flagged.iloc[5]))
        self.assertTrue(scores.flagged.iloc[6])
        self.assertFalse(scores.flagged.iloc[7])  # Strict >, not >=.
        altered = terms.copy()
        altered[~calibration] = 1e15
        self.assertEqual(calibrate(frame, altered, calibration)[1], cut)
        with self.assertRaisesRegex(ValueError, 'No observed'):
            calibrate(frame, terms, np.zeros(len(frame), dtype=bool))

    def test_historical_base_float32_score_is_preserved_exactly(self):
        rng = np.random.default_rng(42)
        training = rng.uniform(0, 1, size=(12, 16)).astype(np.float32)
        features = rng.uniform(0, 1, size=(7, 16))
        features[0] = 0
        saved = {'scaler': StandardScaler().fit(training), 'model': FixedPrediction()}
        before = features.copy()
        x = saved['scaler'].transform(features.astype(np.float32))
        expected = (x - saved['model'].predict(x)) ** 2
        terms, original_score = base_terms(features, saved)
        self.assertEqual(saved['model'].last_input.dtype, np.float32)
        self.assertEqual(original_score.dtype, np.float32)
        np.testing.assert_array_equal(original_score, expected.mean(axis=1))
        np.testing.assert_array_equal(terms, expected.reshape(7, 8, 2).mean(axis=2))
        np.testing.assert_array_equal(features, before)
        self.assertTrue(np.isfinite(original_score[0]))
        frame = pd.DataFrame({'dcs_pass': True}, index=np.arange(7))
        scores, _ = calibrate(frame, terms, np.array([True]*3 + [False]*4), values=original_score)
        np.testing.assert_array_equal(scores.anomaly_score, original_score.astype(float))

    def test_feature_contributions_follow_the_number_of_mean_std_pairs(self):
        features = np.random.default_rng(3).normal(size=(7, 6)).astype(np.float32)
        saved = {'scaler': StandardScaler().fit(features), 'model': FixedPrediction()}
        x = saved['scaler'].transform(features)
        squared = (x - saved['model'].predict(x)) ** 2
        terms, score = base_terms(features, saved)
        self.assertEqual(terms.shape, (7, 3))
        np.testing.assert_array_equal(terms, squared.reshape(7, 3, 2).mean(2))
        np.testing.assert_array_equal(score, squared.mean(1))
        with self.assertRaisesRegex(ValueError, 'mean/std pairs'):
            base_terms(features[:, :5], saved)

    def test_stale_identifiers_splits_counts_or_duplicate_rows_are_rejected(self):
        frame = pd.DataFrame(dict(dataset=['d', 'd'], run_number=[1, 2], ls_number=[1, 1],
                                  label=['good']*2, split=['train', 'test'], dcs_pass=[True]*2,
                                  **{f'N_{i}': [10, 20] for i in range(8)}))
        require_alignment({'old': frame.iloc[::-1]}, frame)
        for column, replacement in [('split', 'test'), ('N_3', 99), ('dcs_pass', False)]:
            changed = frame.copy()
            changed.loc[0, column] = replacement
            with self.assertRaises(ValueError):
                require_alignment({'old': changed}, frame)
        for invalid in (frame.iloc[:1], pd.concat([frame.iloc[:1]]*2)):
            with self.assertRaises(ValueError):
                require_alignment({'old': invalid}, frame)

    def test_window_uses_adjacent_ls_without_labels_and_resets_all_boundaries(self):
        frame = pd.DataFrame(dict(dataset=['d']*9+['other'], run_number=[1]*7+[2]*2+[1],
                                  ls_number=[1, 2, 3, 4, 8, 9, 10, 1, 2, 1],
                                  dcs_pass=[True]*5+[False]*2+[True]*3,
                                  anomaly_score=[0., 10., 20., 30., 100., 1000., 3000., 7., 9., 999.],
                                  label=['good']*10))
        expected = [10., 15., 15., 20., 100., 2000., 2000., 8., 8., 999.]
        np.testing.assert_allclose(window_mean(frame), expected)
        frame['label'] = 'bad'
        np.testing.assert_array_equal(window_mean(frame), expected)
        frame.loc[1, 'anomaly_score'] = np.nan
        self.assertTrue(np.isnan(window_mean(frame)[1]))
        shuffled = frame.sample(frac=1, random_state=17)
        np.testing.assert_allclose(window_mean(shuffled), pd.Series(window_mean(frame), index=frame.index).loc[shuffled.index],
                                   equal_nan=True)
        duplicated_index = frame.copy()
        duplicated_index.index = [0]*len(frame)
        np.testing.assert_allclose(window_mean(duplicated_index), window_mean(frame), equal_nan=True)
        with self.assertRaises(ValueError):
            window_mean(pd.concat([frame.iloc[:1]]*2))

    def test_special_controls_are_not_standard_good_or_count_cohorts_but_d_is_reported(self):
        runs = [101, 102, 103, 104, *SPECIAL_RUNS, 106, 207, 108]
        scores = pd.DataFrame({'run_number': runs, 'split': ['test']*8 + ['train'],
                               'label': ['good']*6 + ['bad', 'bad', 'good'],
                               'dcs_pass': [True, True, True, False, True, True, True, True, True],
                               'minimum_positive_N': [5, 50, 5, 300, 5, 5, 5, 5, 5],
                               'anomaly_score': [.5, 2., np.nan, 8., 4., 4., 3., 1., 9.]})
        scores['flagged'] = pd.array([False, True, None, None, True, True, True, False, True], dtype='boolean')
        rows = {row['cohort']: row for row in summarize(scores)}
        self.assertEqual((rows['test_good_standard']['included_ls'], rows['test_good_standard']['scored_ls']), (3, 2))
        self.assertEqual(rows['test_good_standard']['flagged_fraction'], .5)
        self.assertEqual((rows['N1_10']['included_ls'], rows['N1_10']['scored_ls'], rows['N1_10']['flagged_ls']), (2, 1, 0))
        self.assertEqual(rows['N11_100']['included_ls'], 1)
        self.assertEqual(rows['Ngt100']['included_ls'], 0)
        self.assertEqual((rows['D']['included_ls'], rows['D']['flagged_ls']), (1, 1))
        self.assertEqual(rows['test_bad']['flagged_ls'], 1)
        self.assertEqual(rows['test_bad']['included_ls'], 2)
        self.assertIsNone(rows['A']['flagged_fraction'])


if __name__ == '__main__':
    unittest.main()
