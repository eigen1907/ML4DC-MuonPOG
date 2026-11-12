"""The mean/std baseline and two matched, single-component histogram AEs."""

import copy
import math

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from config import SEED


HIDDEN_LAYERS = (8, 4, 8)
BASE_EPOCHS = 80
EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 1024


def _rows(selection, size):
    selection = np.asarray(selection)
    if selection.dtype == bool:
        if selection.shape != (size,):
            raise ValueError('Expected one training flag per LS')
        selection = np.flatnonzero(selection)
    if (selection.ndim != 1 or not np.issubdtype(selection.dtype, np.integer)
            or not len(selection) or len(np.unique(selection)) != len(selection)
            or (selection < 0).any() or (selection >= size).any()):
        raise ValueError('Expected nonempty unique LS indices')
    return selection


def fit_base(features, train_mask, monitor_indices):
    """Mean/std pairs through an 8→4→8 bottleneck; monitoring never selects epochs."""
    features = np.asarray(features, dtype=np.float32)
    if (features.ndim != 2 or features.shape[1] == 0 or features.shape[1] % 2
            or not np.isfinite(features).all()):
        raise ValueError('Expected a finite LS-by-feature matrix of mean/std pairs')
    train, monitor = _rows(train_mask, len(features)), _rows(monitor_indices, len(features))
    if np.intersect1d(train, monitor).size:
        raise ValueError('Training and monitoring LS must be disjoint')
    scaler = StandardScaler().fit(features[train])
    x, checked = scaler.transform(features[train]), scaler.transform(features[monitor])
    model = MLPRegressor(hidden_layer_sizes=HIDDEN_LAYERS, batch_size=BATCH_SIZE,
                         random_state=SEED, tol=0, n_iter_no_change=BASE_EPOCHS + 1)
    history = {'train': [], 'validation': []}
    for epoch in range(1, BASE_EPOCHS + 1):
        model.partial_fit(x, x)
        history['train'].append(float(np.mean((model.predict(x) - x) ** 2)))
        history['validation'].append(float(np.mean((model.predict(checked) - checked) ** 2)))
        if epoch % 20 == 0:
            print(f'base epoch {epoch:02d}: train {history["train"][-1]:.7g}, '
                  f'monitor {history["validation"][-1]:.7g}', flush=True)
    return {'model': model, 'scaler': scaler, 'hidden_layer_sizes': HIDDEN_LAYERS,
            'seed': SEED, 'train_losses': history['train'],
            'validation_losses': history['validation']}


class HistogramAE(nn.Module):
    """Native normalized bins + log-N + masks → 64 → 4 → 64 → histogram heads.

    DM has exactly one component. Its unused one-component weight heads remain
    present so the preserved dm_k1 checkpoint loads without changing any keys.
    Concentration is learned per ME and shared across LS.
    """
    def __init__(self, bin_sizes, distribution='mse'):
        super().__init__()
        self.bin_sizes = tuple(int(size) for size in bin_sizes)
        if (not self.bin_sizes or min(self.bin_sizes) < 1
                or distribution not in ('mse', 'dm')):
            raise ValueError('Expected positive bin sizes and distribution mse or dm')
        self.distribution = distribution
        self.components = 1
        self.input_features = sum(self.bin_sizes) + 2 * len(self.bin_sizes)
        self.encoder = nn.Sequential(nn.Linear(self.input_features, 64), nn.ReLU(), nn.Linear(64, 4))
        self.decoder = nn.Sequential(nn.Linear(4, 64), nn.ReLU())
        self.shape_probabilities = nn.ModuleList(nn.Linear(64, size) for size in self.bin_sizes)
        self.shape_weights = (nn.ModuleList(nn.Linear(64, 1) for _ in self.bin_sizes)
                              if distribution == 'dm' else None)
        if distribution == 'dm':
            self.log_concentration = nn.Parameter(torch.full((len(self.bin_sizes),), math.log(100.)))
        else:
            self.register_parameter('log_concentration', None)

    def initialize_heads(self, pooled_counts):
        """Pooled training-only shapes, with 0.5 count per native bin."""
        if len(pooled_counts) != len(self.bin_sizes):
            raise ValueError('Expected pooled counts for every ME')
        with torch.no_grad():
            for bins, head, counts in zip(self.bin_sizes, self.shape_probabilities, pooled_counts):
                counts = torch.as_tensor(counts, device=head.weight.device, dtype=torch.float64)
                if counts.shape != (bins,) or not torch.isfinite(counts).all() or (counts < 0).any():
                    raise ValueError('Expected finite nonnegative pooled bin counts')
                counts = counts + .5
                head.bias.copy_((counts / counts.sum()).log().to(head.bias.dtype))
                head.weight.mul_(.1)
            if self.shape_weights is not None:
                for head in self.shape_weights:
                    head.weight.mul_(.1)
                    head.bias.zero_()

    def forward(self, inputs):
        hidden = self.decoder(self.encoder(inputs))
        predictions = []
        for index, head in enumerate(self.shape_probabilities):
            probabilities = head(hidden).unsqueeze(1).softmax(-1).clamp_min(1e-8)
            probabilities = probabilities / probabilities.sum(-1, keepdim=True)
            value = {'probabilities': probabilities,
                     'logits': self.shape_weights[index](hidden) if self.shape_weights is not None
                     else hidden.new_zeros((len(hidden), 1))}
            if self.log_concentration is not None:
                value['concentration'] = self.log_concentration[index].clamp(
                    math.log(1e-3), math.log(1e8)).exp()
            predictions.append(value)
        return predictions


def dm_log_prob(counts, probabilities, concentration):
    """Exact Dirichlet–multinomial log PMF in float64, including combinatorics.

    Counts [..., bins] broadcast against probabilities [..., 1, bins]. N=0
    has exactly zero log probability and contributes no shape gradient.
    """
    probabilities = probabilities.double()
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    counts = torch.as_tensor(counts, device=probabilities.device, dtype=torch.float64)
    if counts.ndim + 1 == probabilities.ndim:
        counts = counts.unsqueeze(-2)
    concentration = torch.as_tensor(concentration, device=probabilities.device, dtype=torch.float64)
    alpha = probabilities * concentration.unsqueeze(-1)
    total, alpha_total = counts.sum(-1), alpha.sum(-1)
    value = (torch.lgamma(total + 1) - torch.lgamma(counts + 1).sum(-1)
             + torch.lgamma(alpha_total) - torch.lgamma(total + alpha_total)
             + (torch.lgamma(counts + alpha) - torch.lgamma(alpha)).sum(-1))
    return torch.where(total > 0, value, torch.zeros_like(value))


def shape_log_prob(counts, prediction):
    return dm_log_prob(counts, prediction['probabilities'], prediction['concentration']).squeeze(-1)


def variance(prediction, totals):
    """DM fraction variance, conditional on N; N=0 has no measured shape.

    This is decoder variance, not exact self-reconstruction error variance.
    """
    p = prediction['probabilities'][:, 0].double()
    p = p / p.sum(-1, keepdim=True)
    totals = torch.as_tensor(totals, device=p.device, dtype=torch.float64)
    concentration = torch.as_tensor(prediction['concentration'], device=p.device, dtype=torch.float64)
    n = totals.clamp_min(1)
    value = p * (1 - p) * ((n + concentration) / (n * (1 + concentration)))[:, None]
    return torch.where(totals[:, None] > 0, value, torch.full_like(value, torch.nan))


def residuals(histograms, predictions):
    """Native-bin mean squared error per ME, optionally divided by DM variance.

    Empty MEs return NaN. Weighted residuals ignore cross-bin covariance and
    are a diagnostic, not chi-square or a probability of bad data.
    """
    if not histograms or len(histograms) != len(predictions):
        raise ValueError('Expected matching histograms and predictions')
    mse, weighted = [], []
    for histogram, prediction in zip(histograms, predictions):
        p = prediction['probabilities'][:, 0].double()
        p = p / p.sum(-1, keepdim=True)
        counts = torch.as_tensor(histogram, device=p.device, dtype=torch.float64)
        totals = counts.sum(-1)
        squared = (counts / totals.clamp_min(1)[:, None] - p) ** 2
        missing = torch.full_like(totals, torch.nan)
        mse.append(torch.where(totals > 0, squared.mean(-1), missing))
        if 'concentration' in prediction:
            value = (squared / variance(prediction, totals).clamp_min(1e-12)).mean(-1)
            weighted.append(torch.where(totals > 0, value, missing))
    result = {'mse': torch.stack(mse, dim=1)}
    if weighted:
        if len(weighted) != len(predictions):
            raise ValueError('All predictions must use the same distribution')
        result['weighted'] = torch.stack(weighted, dim=1)
    return result


def loss(predictions, histograms, distribution):
    """Mean over observed MEs within each LS, then over observed LS."""
    if not histograms or len(histograms) != len(predictions):
        raise ValueError('Expected matching histograms and predictions')
    valid = torch.stack([torch.as_tensor(h, device=p['probabilities'].device).sum(-1) > 0
                         for h, p in zip(histograms, predictions)], dim=1)
    observed = valid.any(1)
    if not observed.any():
        return None
    if distribution == 'mse':
        values = residuals(histograms, predictions)['mse']
    elif distribution == 'dm':
        values = torch.stack([-shape_log_prob(h, p) for h, p in zip(histograms, predictions)], dim=1)
    else:
        raise ValueError('Expected distribution mse or dm')
    per_ls = torch.where(valid, values, 0).sum(1) / valid.sum(1).clamp_min(1)
    return per_ls[observed].mean()


def _histograms(dataset):
    histograms = [np.asarray(h) for h in dataset['histograms']]
    totals = np.asarray(dataset['totals'])
    if (not histograms or totals.ndim != 2 or totals.shape[1] != len(histograms)
            or not np.isfinite(totals).all() or (totals < 0).any()
            or (not np.issubdtype(totals.dtype, np.integer) and (totals != np.floor(totals)).any())):
        raise ValueError('Expected finite nonnegative integer histogram totals')
    for index, hist in enumerate(histograms):
        if (hist.ndim != 2 or len(hist) != len(totals) or hist.shape[1] < 1
                or not np.isfinite(hist).all() or (hist < 0).any()
                or (not np.issubdtype(hist.dtype, np.integer) and (hist != np.floor(hist)).any())
                or not np.array_equal(hist.sum(-1, dtype=np.float64), totals[:, index])):
            raise ValueError('Invalid native-bin histogram or inconsistent totals')
    return histograms, totals


def prepare(dataset, train_mask, scaler=None):
    """Native fractions + training-scaled log1p(N) + nonempty masks."""
    histograms, totals = _histograms(dataset)
    train = _rows(train_mask, len(totals))
    valid = totals > 0
    bins, targets = sum(h.shape[1] for h in histograms), len(histograms)
    values = np.zeros((len(totals), bins + 2 * targets), dtype=np.float32)
    start = 0
    for target, hist in enumerate(histograms):
        stop = start + hist.shape[1]
        np.divide(hist, totals[:, target, None], out=values[:, start:stop], where=valid[:, target, None])
        start = stop
    logs = np.log1p(totals).astype(np.float32)
    if scaler is None:
        scaler = StandardScaler().fit(logs[train])
    values[:, bins:bins + targets] = scaler.transform(logs)
    values[:, bins + targets:] = valid
    return torch.from_numpy(values), scaler


def _monitor_loss(model, inputs, histograms, totals, indices, distribution):
    model.eval()
    total, weight = 0., 0
    with torch.no_grad():
        for first in range(0, len(indices), BATCH_SIZE):
            rows = indices[first:first + BATCH_SIZE]
            value = loss(model(inputs[rows]), [h[rows] for h in histograms], distribution)
            n = int((totals[rows] > 0).any(axis=1).sum())
            if value is not None:
                total += float(value) * n
                weight += n
    if not weight:
        raise ValueError('No nonempty monitoring LS')
    return total / weight


def fit_histogram(dataset, inputs, train_mask, monitor_indices, distribution):
    """Matched K=1 training; the caller supplies Good training/monitoring LS."""
    histograms, totals = _histograms(dataset)
    train, monitor = _rows(train_mask, len(totals)), _rows(monitor_indices, len(totals))
    if np.intersect1d(train, monitor).size:
        raise ValueError('Training and monitoring LS must be disjoint')
    if not (totals[train] > 0).any():
        raise ValueError('No nonempty training LS')
    inputs = torch.as_tensor(inputs, dtype=torch.float32)
    expected = (len(totals), sum(h.shape[1] for h in histograms) + 2 * len(histograms))
    if tuple(inputs.shape) != expected or not torch.isfinite(inputs).all():
        raise ValueError('Invalid prepared histogram inputs')
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = HistogramAE([h.shape[1] for h in histograms], distribution)
    model.initialize_heads([h[train].sum(axis=0, dtype=np.float64) for h in histograms])
    parameters = list(model.parameters())
    groups = parameters if distribution == 'mse' else [
        {'params': [p for p in parameters if p is not model.log_concentration]},
        {'params': [model.log_concentration], 'lr': .01}]
    optimizer = torch.optim.Adam(groups, lr=.001, eps=1e-12)
    history = {'train': [], 'validation': [], 'best_epoch': 0,
               'loss': 'MSE' if distribution == 'mse' else 'DM NLL'}
    best, state, stale = float('inf'), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        shuffled, total, weight = rng.permutation(train), 0., 0
        for first in range(0, len(shuffled), BATCH_SIZE):
            rows = shuffled[first:first + BATCH_SIZE]
            optimizer.zero_grad(set_to_none=True)
            value = loss(model(inputs[rows]), [h[rows] for h in histograms], distribution)
            if value is None:
                continue
            if not torch.isfinite(value):
                raise FloatingPointError(f'Nonfinite {distribution} training loss')
            value.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5.)
            optimizer.step()
            n = int((totals[rows] > 0).any(axis=1).sum())
            total += float(value.detach()) * n
            weight += n
        checked = _monitor_loss(model, inputs, histograms, totals, monitor, distribution)
        if not np.isfinite(checked):
            raise FloatingPointError(f'Nonfinite {distribution} monitoring loss')
        history['train'].append(total / weight)
        history['validation'].append(checked)
        print(f'{distribution} epoch {epoch:02d}: train {total / weight:.7g}, monitor {checked:.7g}', flush=True)
        if checked < best * (1 - 1e-4):
            best, state, stale = checked, copy.deepcopy(model.state_dict()), 0
            history['best_epoch'] = epoch
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(state)
    model.eval()
    return model, history
