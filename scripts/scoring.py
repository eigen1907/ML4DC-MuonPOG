"""Shared residuals, fixed Good calibration and LS context."""

import numpy as np
import pandas as pd
import torch

from config import IDS, WINDOW_SIZE
from models import BATCH_SIZE, residuals


QUANTILE = .975


def mean_score(terms):
    terms = np.asarray(terms, dtype=float)
    if terms.ndim != 2 or np.isinf(terms).any():
        raise ValueError('Expected an LS-by-ME array of finite or missing residuals')
    valid = np.isfinite(terms)
    return np.divide(np.where(valid, terms, 0).sum(1), valid.sum(1),
                     out=np.full(len(terms), np.nan), where=valid.any(1))


def base_terms(features, saved):
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
