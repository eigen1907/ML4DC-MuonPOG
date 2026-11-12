"""Shared networks, likelihoods, scoring and checkpoint loading; no fitting."""

import importlib
import json
import math
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn

from scripts.models.config import DM_AE, EXPERIMENTS, FEATURE_AE, HIST_AE, IDS, OUT, WINDOW_AE, WINDOW_SIZE

HIDDEN_LAYERS = (8, 4, 8)
FEATURE_EPOCHS = 80
EPOCHS = 60
PATIENCE = 8
BATCH_SIZE = 1024


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


def model_spec(name):
    """Select one explicit model module by its experiment or module name."""
    name = name.removeprefix('model_')
    if name not in EXPERIMENTS:
        raise ValueError(f'Unknown model: {name}')
    return importlib.import_module(f'scripts.models.model_{name}')


def load_checkpoint(path):
    """Read a saved model or window pointer, including preserved checkpoints."""
    path = Path(path)
    if path.is_dir():
        files = [path/name for name in ('model.pkl', 'model.pt', 'source.json') if (path/name).is_file()]
        if len(files) != 1:
            raise ValueError(f'Expected one model checkpoint in {path}')
        path = files[0]
    if path.suffix == '.json':
        pointer = json.loads(path.read_text())
        spec = model_spec(pointer.get('experiment', WINDOW_AE))
        if spec.KIND != 'window':
            raise ValueError('Expected a window-model source pointer')
        source = (path.parent/str(pointer['source']) if 'source' in pointer else
                  path.parent.parent/pointer.get('source_model', spec.SOURCE))
        if source.resolve() == path.resolve():
            raise ValueError('A window checkpoint cannot refer to itself')
        source_spec, saved, source_file = load_checkpoint(source)
        if source_spec.NAME != spec.SOURCE:
            raise ValueError(f'{spec.NAME} requires source model {spec.SOURCE}')
        saved = {key: value for key, value in saved.items()
                 if key not in ('experiment', 'threshold', 'quantile')}
        saved.update(pointer, source_file=str(source_file), source_model=source_spec.NAME)
        return spec, saved, path
    if path.suffix == '.pkl':
        saved = pickle.loads(path.read_bytes())
        if 'signature' not in saved:
            raise ValueError('Expected a DCS-trained feature checkpoint')
        inferred = FEATURE_AE
    elif path.suffix == '.pt':
        saved = torch.load(path, map_location='cpu', weights_only=False)
        inferred = DM_AE if 'log_concentration' in saved['state_dict'] else HIST_AE
    else:
        raise ValueError(f'Expected a .pkl, .pt or source.json checkpoint: {path}')
    spec = model_spec(saved.get('experiment', path.parent.name if path.parent.name in EXPERIMENTS else inferred))
    if (path.suffix == '.pt' and 'run_split' not in saved and 'experiment' not in saved
            and path.resolve() == (OUT/'model'/spec.NAME/'model.pt').resolve()):
        # Preserved histogram checkpoints predate embedded run roles.
        roles = pd.read_parquet(OUT/'predict'/spec.NAME/'scores.parquet',
                                columns=['run_number', 'split']).drop_duplicates()
        numbers = pd.to_numeric(roles.run_number, errors='coerce').to_numpy()
        if (roles.empty or not np.isfinite(numbers).all() or (numbers <= 0).any()
                or (numbers != np.floor(numbers)).any()
                or not roles.split.isin(('train', 'validation', 'test')).all()):
            raise ValueError('Invalid saved run roles for the legacy histogram checkpoint')
        roles = roles.assign(run_number=numbers.astype(np.int64))
        if roles.run_number.duplicated().any():
            raise ValueError('A saved run has multiple train/validation/test roles')
        saved['run_split'] = {role: sorted(roles.loc[roles.split.eq(role), 'run_number'].tolist())
                              for role in ('train', 'validation', 'test')}
    return spec, saved, path


def restore_model(spec, saved):
    """Restore the network without fitting or changing its parameters."""
    if spec.KIND == 'feature':
        return saved['model']
    model = spec.build_model(saved['bin_sizes'])
    model.load_state_dict(saved['state_dict'])
    model.eval()
    return model


def threshold_for(path, saved, spec):
    """Use the checkpoint cut, with the historical displayed cut as fallback."""
    metrics = OUT/'predict'/spec.NAME/'metrics.json'
    filename = 'source.json' if spec.KIND == 'window' else 'model.pkl' if spec.KIND == 'feature' else 'model.pt'
    canonical = OUT/'model'/spec.NAME/filename
    path = Path(path)/filename if Path(path).is_dir() else Path(path)
    if saved.get('experiment') == spec.NAME and 'threshold' in saved:
        threshold = saved['threshold']
    elif path.resolve() == canonical.resolve() and metrics.exists():
        # Only the retained project checkpoints may use their published cuts.
        threshold = json.loads(metrics.read_text())['threshold']
    else:
        raise FileNotFoundError(f'No calibration threshold in {path} or {metrics}')
    threshold = float(threshold)
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('Calibration threshold must be positive and finite')
    return threshold


QUANTILE = .975


def mean_score(terms):
    terms = np.asarray(terms, dtype=float)
    if terms.ndim != 2 or np.isinf(terms).any():
        raise ValueError('Expected an LS-by-ME array of finite or missing residuals')
    valid = np.isfinite(terms)
    return np.divide(np.where(valid, terms, 0).sum(1), valid.sum(1),
                     out=np.full(len(terms), np.nan), where=valid.any(1))


def feature_terms(features, saved):
    """Preserve historical feature scaling, empty shapes, and equal ME weights."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not features.shape[1] or features.shape[1] % 2:
        raise ValueError('Expected an LS-by-feature matrix of mean/std pairs')
    x = saved['scaler'].transform(features)
    errors = (x - saved['model'].predict(x)) ** 2
    return errors.reshape(len(errors), features.shape[1] // 2, 2).mean(2), errors.mean(1)


def histogram_terms(model, inputs, histograms, batch_size=BATCH_SIZE):
    """Compute MSE and optional variance-weighted residuals in one decoder pass."""
    model.eval()
    collected = {}
    with torch.no_grad():
        for first in range(0, len(inputs), batch_size):
            stop = first + batch_size
            result = residuals([h[first:stop] for h in histograms], model(inputs[first:stop]))
            for kind, values in result.items():
                collected.setdefault(kind, []).append(values.numpy())
    return {key: np.concatenate(value) for key, value in collected.items()}


def calibrate(frame, terms, calibration, values=None):
    """Set one fixed cut on separate Good calibration LS, retaining all rows."""
    terms = np.asarray(terms)
    if terms.ndim != 2 or len(terms) != len(frame):
        raise ValueError('Expected one row of ME residuals per LS')
    values = mean_score(terms) if values is None else np.asarray(values, dtype=float)
    calibration = np.asarray(calibration)
    if calibration.shape != (len(frame),) or calibration.dtype != bool or values.shape != (len(frame),):
        raise ValueError('Expected one calibration flag and score per LS')
    reference = values[calibration & np.isfinite(values)]
    if not len(reference):
        raise ValueError('No observed Good calibration scores')
    cut = float(np.quantile(reference, QUANTILE))
    if not np.isfinite(cut) or cut <= 0:
        raise ValueError('Calibration cut must be positive and finite')
    scores = frame.copy()
    scores['anomaly_score'] = values
    for target in range(terms.shape[1]):
        scores[f'term_{target}'] = terms[:, target]
    scores['flagged'] = pd.array(np.where(frame.dcs_pass & np.isfinite(values), values > cut, None), dtype='boolean')
    return scores, cut


def window_mean(scores):
    """Offline centered five-LS mean; reset at dataset/run/gap/DCS boundaries.

    Uses up to two future LS. A missing current score stays unscored even
    when observed neighbours contribute context. Labels never define windows.
    """
    if scores.duplicated(IDS).any():
        raise ValueError('Duplicate cached LS identifiers')
    scores = scores.reset_index(drop=True)
    result = np.full(len(scores), np.nan)
    for _, group in scores.groupby(['dataset', 'run_number'], sort=False):
        group = group.sort_values('ls_number')
        breaks = group.ls_number.diff().ne(1) | group.dcs_pass.ne(group.dcs_pass.shift())
        for _, segment in group.groupby(breaks.cumsum(), sort=False):
            values = segment.anomaly_score.rolling(WINDOW_SIZE, center=True, min_periods=1).mean()
            result[segment.index] = values.where(segment.anomaly_score.notna())
    return result
