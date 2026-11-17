"""Checkpoint I/O, ME-error aggregation and Good-only threshold calibration."""

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.models.config import EXPERIMENTS

NORMALIZATION_QUANTILE = .975
THRESHOLD_QUANTILE = .99


def model_spec(name):
    name = name.removeprefix('model_')
    if name not in EXPERIMENTS:
        raise ValueError(f'Unknown model: {name}')
    return importlib.import_module(f'scripts.models.model_{name}')


def load_checkpoint(path):
    path = Path(path)
    if path.is_dir():
        path = path / 'model.pt'
    saved = torch.load(path, map_location='cpu', weights_only=False)
    spec = model_spec(saved['experiment'])
    if spec.KIND == 'gmm' and (saved['settings'].get('loss') != spec.SETTINGS['loss']
                              or saved['settings'].get('loss_normalization') != spec.SETTINGS.get('loss_normalization')):
        raise ValueError('This GMM checkpoint uses the previous loss/score; keep its archived code and results')
    return spec, saved, path


def restore_model(spec, saved):
    model = spec.build_model(saved['input_dim'], saved['settings'])
    model.load_state_dict(saved['state_dict'])
    return model.eval()


def threshold_for(path, saved, spec):
    value = float(saved['threshold'])
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Calibration threshold must be positive and finite')
    return value


def mean_score(terms):
    terms = np.asarray(terms, dtype=float)
    if terms.ndim != 2 or np.isinf(terms).any():
        raise ValueError('Expected finite or missing LS-by-ME residuals')
    valid = np.isfinite(terms)
    return np.divide(np.where(valid, terms, 0).sum(1), valid.sum(1),
                     out=np.full(len(terms), np.nan), where=valid.any(1))


def normalized_max(terms, scales):
    terms, scales = np.asarray(terms, dtype=float), np.asarray(scales, dtype=float)
    if (terms.ndim != 2 or scales.shape != (terms.shape[1],)
            or not np.isfinite(scales).all() or (scales <= 0).any()
            or np.isinf(terms).any() or (terms[np.isfinite(terms)] < 0).any()):
        raise ValueError('Expected nonnegative ME errors and positive normal-error scales')
    valid = np.isfinite(terms)
    values = np.max(np.where(valid, terms/scales, -np.inf), axis=1)
    values[~valid.any(1)] = np.nan
    return values


def score_terms(terms, saved):
    mode = saved.get('score_mode')
    if mode == 'mean':
        return mean_score(terms)
    if mode not in ('deviance_max', 'loss_norm_max', 'norm_deviance_max') or not np.all(np.asarray(saved['me_scales']) == 1):
        raise ValueError('GMM scoring uses its loss errors without another ME normalization')
    return normalized_max(terms, saved['me_scales'])


def calibrate(frame, terms, calibration, scales=None, *, quantile=THRESHOLD_QUANTILE):
    if (isinstance(quantile, bool) or not np.isfinite(quantile) or not 0 < quantile < 1):
        raise ValueError('Calibration quantile must be finite and in (0, 1)')
    values = mean_score(terms) if scales is None else normalized_max(terms, scales)
    calibration = np.asarray(calibration)
    if calibration.dtype != bool or calibration.shape != values.shape or len(frame) != len(values):
        raise ValueError('Expected one calibration flag per LS')
    reference = values[calibration & np.isfinite(values)]
    if not len(reference):
        raise ValueError('No finite Good calibration scores')
    cut = float(np.quantile(reference, quantile))
    if not np.isfinite(cut) or cut <= 0:
        raise ValueError('Calibration threshold must be positive and finite')
    scores = frame.assign(anomaly_score=values)
    for me in range(terms.shape[1]):
        scores[f'term_{me}'] = terms[:, me]
    scores['flagged'] = pd.array(np.where(np.isfinite(values), values > cut, None), dtype='boolean')
    return scores, cut
