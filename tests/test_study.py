"""Tiny five-stage integration: selection, frozen baseline, raw scores and context."""

from contextlib import ExitStack, redirect_stdout
from io import StringIO
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from data import IDS, MODEL_MES, _moments
import models
from scoring import histogram_terms, mean_score, window_mean
import study


def synthetic_dataset():
    rng = np.random.default_rng(17)
    runs = [1]*8+[run for run in range(2, 7) for _ in range(4)]
    sizes = [0, 1, 100, 5000, 5001, 10000, 20, 200]+[100]*20
    histograms = [np.array([rng.multinomial(n, [.2, .5, .3]) for n in sizes], dtype=np.int32)
                  for _ in range(8)]
    totals = np.column_stack([h.sum(1) for h in histograms])
    axes = [dict(low=0., high=3., edges=np.arange(4)) for _ in histograms]
    features = np.column_stack([_moments(h, 0, 3, eta=i == 1) for i, h in enumerate(histograms)])
    frame = pd.DataFrame(dict(dataset=['/Muon0/Run2025B-PromptReco-v1/DQMIO']*28,
        run_number=runs, ls_number=list(range(1, 9))+list(range(1, 5))*5, era=['2025B']*28,
        label=['good']*20+['bad_well_known']*4+['bad_ambiguous']*4,
        split=['train']*8+['validation']*8+['test']*12,
        dcs_pass=[True]*7+[False]+[True]*20, phi_entries=sizes))
    return dict(frame=frame, histograms=histograms, totals=totals, root_entries=totals.copy(),
                features=features, axes=axes, me_names=list(MODEL_MES),
                run_split={'train': [1], 'validation': [2, 3], 'test': [4, 5, 6]})


class StudyTests(unittest.TestCase):
    def test_main_fits_dcs_only_and_writes_five_views_without_changing_historical_model(self):
        data = synthetic_dataset()
        train = np.array([True]*7+[False]*21)
        monitor = np.arange(8, 12)
        original_cwd, original_threads = Path.cwd(), torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
                stack.enter_context(redirect_stdout(StringIO()))
                os.chdir(temporary)
                stack.enter_context(patch.object(models, 'BASE_EPOCHS', 1))
                stack.enter_context(patch.object(models, 'EPOCHS', 1))
                stack.enter_context(warnings.catch_warnings())
                warnings.filterwarnings('ignore', message='Got `batch_size`')
                historical = models.fit_base(data['features'], np.arange(8), monitor)
                historical.update(features=[f'{me}__{kind}' for me in MODEL_MES for kind in ('mean', 'std')],
                                  run_split=data['run_split'])
                path = Path('output/model')/study.STAGES[0]/'model.pkl'
                path.parent.mkdir(parents=True)
                frozen = pickle.dumps(historical)
                path.write_bytes(frozen)
                metadata = Path('data'); metadata.mkdir(parents=True)
                reference = data['frame'].iloc[20:24][IDS].copy()
                reference['anomaly_score'], reference['threshold'] = 2., 1.
                reference.to_csv(metadata/'known_bad_reference.csv', index=False)
                stack.enter_context(patch.object(study, 'OUT', Path('output')))
                stack.enter_context(patch.object(study, 'load_dataset', return_value=data))
                stack.enter_context(patch.object(study, 'fingerprint', return_value='synthetic-28-ls'))
                stack.enter_context(patch.object(study, 'validation_roles', return_value={
                    'monitor_runs': [2], 'calibration_runs': [3]}))
                stack.enter_context(patch.object(sys, 'argv', ['study.py']))
                stack.enter_context(patch.multiple(study, plot_stage=Mock(), plot_comparison=Mock(), plot_retention=Mock()))
                stack.enter_context(patch('diagnostics.write_diagnostics'))
                for method in ('set_num_threads', 'set_num_interop_threads', 'use_deterministic_algorithms'):
                    stack.enter_context(patch.object(study.torch, method))
                feature_fit = stack.enter_context(patch.object(study, 'fit_base', wraps=models.fit_base))
                histogram_fit = stack.enter_context(patch.object(study, 'fit_histogram', wraps=models.fit_histogram))
                study.main()
                self.assertEqual(feature_fit.call_count, 1)
                self.assertEqual(histogram_fit.call_count, 2)
                np.testing.assert_array_equal(feature_fit.call_args.args[1], train)
                self.assertTrue(feature_fit.call_args.args[1][0])  # Empty DCS-ready LS stays in Base training.
                self.assertEqual([call.args[-1] for call in histogram_fit.call_args_list], ['mse', 'dm'])
                for call in histogram_fit.call_args_list:
                    np.testing.assert_array_equal(call.args[2], train)
                    np.testing.assert_array_equal(np.sort(call.args[3]), monitor)
                self.assertEqual(path.read_bytes(), frozen)
                frames = {name: pd.read_parquet(Path('output/predict')/name/'scores.parquet') for name in study.STAGES}
                self.assertEqual(len(frames), 5)
                for scores in frames.values():
                    pd.testing.assert_frame_equal(scores[IDS], data['frame'][IDS])
                    self.assertTrue(pd.isna(scores.flagged.iloc[7]))  # Retained, DCS-off row.
                self.assertTrue(np.isfinite(frames[study.STAGES[0]].anomaly_score.iloc[0]))
                for name in study.STAGES[2:]:
                    self.assertTrue(np.isnan(frames[name].anomaly_score.iloc[0]))
                dm_path = Path('output/model')/study.STAGES[3]/'model.pt'
                saved = torch.load(dm_path, map_location='cpu', weights_only=False)
                np.testing.assert_allclose(saved['scaler'].mean_, np.log1p(data['totals'][train]).astype(np.float32).mean(0), rtol=1e-6)
                model = models.HistogramAE(saved['bin_sizes'], 'dm')
                model.load_state_dict(saved['state_dict'], strict=True)
                inputs, _ = models.prepare(data, train, saved['scaler'])
                expected = mean_score(histogram_terms(model, inputs, data['histograms'])['weighted'])
                np.testing.assert_array_equal(frames[study.STAGES[3]].anomaly_score, expected)
                np.testing.assert_array_equal(frames[study.STAGES[4]].anomaly_score, window_mean(frames[study.STAGES[3]]))
                source = json.loads((Path('output/model')/study.STAGES[4]/'source.json').read_text())
                self.assertEqual(source['source_model'], study.STAGES[3])
                self.assertFalse(list((Path('output/model')/study.STAGES[4]).glob('*.pt')))
                audit = pd.read_csv('output/predict/comparison/known_bad_retention.csv')
                self.assertTrue((audit.protected_ls == 4).all())
                protocol = json.loads(Path('output/predict/protocol.json').read_text())
                self.assertEqual(protocol['training_ls'], 7)
                self.assertEqual(protocol['calibration_ls'], 4)
                self.assertEqual(protocol['bin_sizes'], [3]*len(MODEL_MES))

                # A normal resume must load all checkpoints and reproduce every score.
                feature_fit.side_effect = AssertionError('Unexpected feature-AE refit')
                histogram_fit.side_effect = AssertionError('Unexpected histogram-AE refit')
                study.main()
                for name, expected_scores in frames.items():
                    actual = pd.read_parquet(Path('output/predict')/name/'scores.parquet')
                    pd.testing.assert_frame_equal(actual, expected_scores, check_exact=True)

                self.assertEqual(path.read_bytes(), frozen)

                # A checkout without the historical binary can reproduce its recipe.
                path.unlink()
                feature_fit.side_effect = None
                study.main()
                refitted = pickle.loads(path.read_bytes())
                self.assertEqual(refitted['origin'], 'Historical recipe refit')
                protocol = json.loads(Path('output/predict/protocol.json').read_text())
                self.assertEqual(protocol['historical_base'], refitted['origin'])
                for name, expected_scores in frames.items():
                    actual = pd.read_parquet(Path('output/predict')/name/'scores.parquet')
                    pd.testing.assert_frame_equal(actual, expected_scores, check_exact=True)

                # Equal bin counts with different physical binning cannot reuse a fit.
                changed_path = Path('output/model')/study.STAGES[2]/'model.pt'
                changed = torch.load(changed_path, map_location='cpu', weights_only=False)
                changed['axes'][0]['edges'] = changed['axes'][0]['edges'] + 1
                torch.save(changed, changed_path)
                with self.assertRaisesRegex(ValueError, 'axes'):
                    study.main()
        finally:
            os.chdir(original_cwd)
            torch.set_num_threads(original_threads)


if __name__ == '__main__':
    unittest.main()
