"""Small numerical controls for the retained baseline and native-bin AEs."""

import copy
import io
from pathlib import Path
import pickle
import sys
import unittest
from unittest.mock import patch
import warnings

import numpy as np
from scipy.stats import dirichlet_multinomial
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import models

def prediction(p, rows=1, concentration=9.):
    return {'probabilities': torch.tensor(p, dtype=torch.float64)[None, None].repeat(rows, 1, 1),
            'logits': torch.zeros(rows, 1, dtype=torch.float64),
            'concentration': torch.tensor(concentration, dtype=torch.float64)}

def dataset():
    rng = np.random.default_rng(17)
    histograms = [rng.integers(0, 5, size=(24, n), dtype=np.int32) for n in (3, 5, 4, 6, 2, 3, 4, 5)]
    for h in histograms:
        h[0] = 0
    return {'histograms': histograms, 'totals': np.column_stack([h.sum(1) for h in histograms])}

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_dm_logpmf_is_exact_and_empty_shape_has_no_gradient(self):
        counts = torch.tensor([[2, 0, 3], [0, 0, 0]])
        p = torch.tensor([[[.2, .3, .5]], [[.1, .5, .4]]], dtype=torch.float64, requires_grad=True)
        actual = models.dm_log_prob(counts, p, 7.)
        expected = dirichlet_multinomial.logpmf([2, 0, 3], 7*np.array([.2, .3, .5]), 5)
        self.assertEqual(actual.dtype, torch.float64)
        self.assertAlmostEqual(actual[0, 0].item(), expected, places=12)
        self.assertEqual(actual[1, 0].item(), 0.)
        (-actual.sum()).backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        torch.testing.assert_close(p.grad[1], torch.zeros_like(p.grad[1]), rtol=0, atol=0)

    def test_dm_fraction_variance_matches_enumerated_distribution(self):
        p, n, k = np.array([.2, .3, .5]), 4, 9.
        counts = np.array([(a, b, n-a-b) for a in range(n+1) for b in range(n-a+1)])
        probabilities = dirichlet_multinomial.pmf(counts, k*p, n)
        self.assertAlmostEqual(probabilities.sum(), 1., places=12)
        expected = (probabilities[:, None] * (counts/n - p)**2).sum(0)
        actual = models.variance(prediction(p, rows=2, concentration=k), [n, 0])
        np.testing.assert_allclose(actual[0].numpy(), expected, rtol=1e-12, atol=1e-14)
        self.assertTrue(torch.isnan(actual[1]).all())

    def test_missing_mes_and_ls_are_masked_and_loss_balances_mes(self):
        histograms = [np.array([[1, 0], [0, 0], [0, 0]]),
                      np.array([[1, 1], [3, 0], [0, 0]])]
        preds = [prediction([.5, .5], rows=3) for _ in histograms]
        errors = models.residuals(histograms, preds)
        expected = torch.tensor([[.25, 0.], [torch.nan, .25], [torch.nan, torch.nan]], dtype=torch.float64)
        torch.testing.assert_close(errors['mse'], expected, equal_nan=True)
        torch.testing.assert_close(torch.isnan(errors['weighted']), torch.isnan(expected))
        self.assertAlmostEqual(models.loss(preds, histograms, 'mse').item(), .1875)
        nll0 = -dirichlet_multinomial.logpmf([1, 0], [4.5, 4.5], 1)
        nll1 = -dirichlet_multinomial.logpmf([1, 1], [4.5, 4.5], 2)
        nll2 = -dirichlet_multinomial.logpmf([3, 0], [4.5, 4.5], 3)
        self.assertAlmostEqual(models.loss(preds, histograms, 'dm').item(), ((nll0+nll1)/2+nll2)/2, places=12)
        for kind in ('mse', 'dm'):
            self.assertIsNone(models.loss(preds, [np.zeros_like(h) for h in histograms], kind))

    def test_prepare_fits_only_train_and_retains_native_bins_and_missing_masks(self):
        data = dataset()
        train = np.arange(8)
        values, scaler = models.prepare(data, train)
        changed = copy.deepcopy(data)
        for h in changed['histograms']:
            h[8:] *= 100
        changed['totals'][8:] *= 100
        other, other_scaler = models.prepare(changed, train)
        np.testing.assert_array_equal(scaler.mean_, other_scaler.mean_)
        np.testing.assert_array_equal(scaler.scale_, other_scaler.scale_)
        torch.testing.assert_close(values[train], other[train], rtol=0, atol=0)
        np.testing.assert_allclose(scaler.mean_, np.log1p(data['totals'][train]).astype(np.float32).mean(0), rtol=1e-6)
        self.assertEqual(values.dtype, torch.float32)
        bins = sum(h.shape[1] for h in data['histograms'])
        self.assertEqual(values.shape, (24, bins+16))
        torch.testing.assert_close(values[0, :bins], torch.zeros(bins), rtol=0, atol=0)
        torch.testing.assert_close(values[:, bins+8:], torch.tensor(data['totals'] > 0).float(), rtol=0, atol=0)
        transformed, reused = models.prepare(changed, train, scaler)
        self.assertIs(reused, scaler)
        torch.testing.assert_close(transformed, other, rtol=0, atol=0)
        changed['totals'][0, 0] = 1
        with self.assertRaises(ValueError):
            models.prepare(changed, train)

    def test_k1_checkpoint_keys_and_trunk_initialization_are_preserved(self):
        sizes = (50, 100, 100, 500, 40, 40, 30, 120)
        torch.manual_seed(42)
        mse = models.HistogramAE(sizes, 'mse')
        torch.manual_seed(42)
        dm = models.HistogramAE(sizes, 'dm')
        self.assertEqual(dm.input_features, 996)
        for name, value in mse.encoder.state_dict().items():
            torch.testing.assert_close(value, dm.encoder.state_dict()[name], rtol=0, atol=0)
        keys = {'log_concentration', 'encoder.0.weight', 'encoder.0.bias', 'encoder.2.weight',
                'encoder.2.bias', 'decoder.0.weight', 'decoder.0.bias'}
        keys.update(f'{head}.{i}.{kind}' for head in ('shape_probabilities', 'shape_weights')
                    for i in range(8) for kind in ('weight', 'bias'))
        self.assertEqual(set(dm.state_dict()), keys)
        buffer = io.BytesIO()
        torch.save(dm.state_dict(), buffer)
        buffer.seek(0)
        restored = models.HistogramAE(sizes, 'dm')
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        x = torch.randn(2, 996)
        for expected, actual in zip(dm(x), restored(x)):
            self.assertEqual(actual['probabilities'].shape[1], 1)
            for name in expected:
                torch.testing.assert_close(expected[name], actual[name], rtol=0, atol=0)
        checkpoint = ROOT / 'output/model/04_ae_dm_k1_dcs/model.pt'
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
            preserved = models.HistogramAE(saved['bin_sizes'], 'dm')
            preserved.load_state_dict(saved['state_dict'], strict=True)

    def test_one_epoch_histogram_fit_is_independent_of_test_data(self):
        data = dataset()
        train, monitor = np.arange(8), np.arange(8, 16)
        changed = copy.deepcopy(data)
        for h in changed['histograms']:
            h[16:] *= 1000
        changed['totals'][16:] *= 1000
        for kind in ('mse', 'dm'):
            with patch.object(models, 'EPOCHS', 1):
                inputs, _ = models.prepare(data, train)
                original, history = models.fit_histogram(data, inputs, train, monitor, kind)
                inputs, _ = models.prepare(changed, train)
                other, other_history = models.fit_histogram(changed, inputs, train, monitor, kind)
            self.assertEqual(history, other_history)
            self.assertEqual(history['best_epoch'], 1)
            for name, value in original.state_dict().items():
                torch.testing.assert_close(value, other.state_dict()[name], rtol=0, atol=0)
            for prediction in original(inputs):
                p = prediction['probabilities'][:, 0]
                torch.testing.assert_close(p.sum(1), torch.ones(len(inputs)))
            with self.assertRaises(ValueError):
                models.fit_histogram(data, inputs, train, [0, 1], kind)

    def test_base_matches_original_loop_and_monitor_does_not_change_weights(self):
        features = np.random.default_rng(17).normal(size=(80, 16)).astype(np.float32)
        train, monitor = np.arange(48), np.arange(48, 64)
        with warnings.catch_warnings(), patch.object(models, 'BASE_EPOCHS', 3):
            warnings.filterwarnings('ignore', message='Got `batch_size`')
            saved = models.fit_base(features, train, monitor)
            changed = features.copy()
            changed[48:] *= -100
            other = models.fit_base(changed, train, monitor)
            scaler = StandardScaler().fit(features[train])
            x, checked = scaler.transform(features[train]), scaler.transform(features[monitor])
            original = MLPRegressor(hidden_layer_sizes=(8, 4, 8), batch_size=1024,
                                    random_state=42, tol=0, n_iter_no_change=4)
            train_loss, validation_loss = [], []
            for _ in range(3):
                original.partial_fit(x, x)
                train_loss.append(float(np.mean((original.predict(x)-x)**2)))
                validation_loss.append(float(np.mean((original.predict(checked)-checked)**2)))
        np.testing.assert_array_equal(saved['train_losses'], train_loss)
        np.testing.assert_array_equal(saved['validation_losses'], validation_loss)
        for actual, expected, holdout in zip(saved['model'].coefs_, original.coefs_, other['model'].coefs_):
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual, holdout)
        restored = pickle.loads(pickle.dumps(saved))
        np.testing.assert_array_equal(restored['model'].predict(checked), original.predict(checked))

    def test_feature_ae_supports_a_changed_me_list_without_changing_its_bottleneck(self):
        features = np.random.default_rng(9).normal(size=(20, 6))
        with warnings.catch_warnings(), patch.object(models, 'BASE_EPOCHS', 1):
            warnings.filterwarnings('ignore', message='Got `batch_size`')
            saved = models.fit_base(features, np.arange(12), np.arange(12, 20))
        self.assertEqual(saved['model'].hidden_layer_sizes, (8, 4, 8))
        self.assertEqual(saved['model'].predict(features[:2]).shape, (2, 6))
        for invalid in (features[:, :5], features[:, :0], features[:, 0]):
            with self.assertRaisesRegex(ValueError, 'mean/std pairs'):
                models.fit_base(invalid, np.arange(12), np.arange(12, 20))

if __name__ == '__main__':
    unittest.main()
