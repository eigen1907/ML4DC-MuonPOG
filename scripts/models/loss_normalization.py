"""Train-only normalization and the shared native-histogram residual error."""

import numpy as np


SQUARED_PROBABILITY_FLOOR = 1e-12
DEVIANCE_SCALE_FLOOR = 1e-6


def histogram_error(residual, scale=1):
    """Mean normalized squared native-bin residual, for NumPy or PyTorch.

    Callers supply the same fixed Train scales during learning and scoring.
    Keeping tensor operations intact preserves PyTorch's gradient graph.
    """
    return (residual**2/scale).mean(-1)


def _probability_batch(values, bins=None):
    values = np.asarray(values, dtype=np.float64)
    if (values.ndim != 2 or not values.shape[1]
            or (bins is not None and values.shape[1] != bins)
            or not np.isfinite(values).all() or (values < 0).any()
            or not np.allclose(values.sum(1), 1, rtol=1e-5, atol=1e-7)):
        raise ValueError('Expected observed normalized Prefit histograms with fixed native bins')
    return values


def fit_me_loss_reference(probability_batches, *, quantile=.975, floor_fraction=.01):
    """Fit one ME's scales from a replayable Good-Train batch factory.

    Every observed training histogram contributes equally. The first pass
    streams its arithmetic mean; the second retains squared deviations for
    exact quantiles, one ME at a time. These are normal shape-variation scales,
    not baseline AE reconstruction errors or counting-uncertainty variances.
    """
    if (not callable(probability_batches) or isinstance(quantile, bool)
            or not np.isfinite(quantile) or not 0 < quantile < 1
            or isinstance(floor_fraction, bool) or not np.isfinite(floor_fraction)
            or floor_fraction <= 0):
        raise ValueError('Expected a replayable batch factory, a quantile in (0, 1), and a positive floor fraction')
    total, observed = None, 0
    for batch in probability_batches():
        values = _probability_batch(batch, None if total is None else len(total))
        if total is None:
            total = np.zeros(values.shape[1], dtype=np.float64)
        total += values.sum(0)
        observed += len(values)
    if not observed:
        raise ValueError('Every ME needs observed Good-Train Prefit histograms')
    mean = total/observed
    squared = np.empty((observed, len(mean)), dtype=np.float64)
    first = 0
    for batch in probability_batches():
        values = _probability_batch(batch, len(mean))
        last = first + len(values)
        if last > observed:
            raise ValueError('Good-Train batch factory changed its observation count on replay')
        squared[first:last] = values-mean
        np.square(squared[first:last], out=squared[first:last])
        first = last
    if first != observed:
        raise ValueError('Good-Train batch factory changed its observation count on replay')
    raw_me_scale = float(np.quantile(squared.mean(1), quantile))
    me_scale = max(raw_me_scale, SQUARED_PROBABILITY_FLOOR)
    raw_bins = np.quantile(squared, quantile, axis=0, overwrite_input=True)
    floor = max(SQUARED_PROBABILITY_FLOOR, float(floor_fraction*me_scale))
    if not np.isfinite(raw_bins).all() or not np.isfinite(me_scale) or not np.isfinite(floor):
        raise ValueError('Normal Prefit variation produced nonfinite loss scales')
    return dict(mean=mean, me_scale=me_scale, bin_scales=np.maximum(raw_bins, floor),
                observed=int(observed), native_bins=int(len(mean)),
                quantile=float(quantile), floor_fraction=float(floor_fraction), floor=floor,
                numerical_floor=SQUARED_PROBABILITY_FLOOR,
                me_scale_raw=raw_me_scale, me_floored=bool(raw_me_scale < SQUARED_PROBABILITY_FLOOR),
                floored_bins=int((raw_bins < floor).sum()),
                reference='Good Train mean Prefit histogram', estimator='squared_deviation_quantile')


def _deviance_batch(values, bins=None):
    values = np.asarray(values)
    if (values.ndim != 2 or not values.shape[1]
            or (bins is not None and values.shape[1] != bins)
            or not np.isfinite(values).all() or (values < 0).any()
            or (values > np.finfo(np.float32).max).any()):
        raise ValueError('Expected finite nonnegative per-bin deviance contributions')
    return values.astype(np.float32, copy=False)


def fit_deviance_reference(deviance_batches, *, observed=None, quantile=.975,
                           floor_fraction=.01):
    """Freeze bin scales, then a ME scale, from one ME's Good-Train errors.

    Batches contain per-bin deviance from the frozen 03 predictor, not Input
    bin probabilities or squared residuals. Supplying the observation count
    permits one teacher-inference pass. Otherwise the factory is replayed
    after counting rows. A float32 buffer is retained for this ME only.
    """
    if (not callable(deviance_batches) or isinstance(quantile, bool)
            or not np.isfinite(quantile) or not 0 < quantile < 1
            or isinstance(floor_fraction, bool) or not np.isfinite(floor_fraction)
            or floor_fraction <= 0 or isinstance(observed, bool)
            or (observed is not None and (not isinstance(observed, (int, np.integer)) or observed < 1))):
        raise ValueError('Expected a batch factory, positive row count and floor, and a quantile in (0, 1)')
    bins = None
    if observed is None:
        observed = 0
        for batch in deviance_batches():
            values = _deviance_batch(batch, bins)
            bins = values.shape[1]
            observed += len(values)
    if not observed:
        raise ValueError('Every ME needs observed Good-Train deviance contributions')
    values, first = None, 0
    for batch in deviance_batches():
        batch = _deviance_batch(batch, bins)
        if values is None:
            bins = batch.shape[1]
            values = np.empty((observed, bins), dtype=np.float32)
        last = first+len(batch)
        if last > observed:
            raise ValueError('Good-Train deviance factory changed its observation count')
        values[first:last] = batch
        first = last
    if first != observed:
        raise ValueError('Good-Train deviance factory changed its observation count')
    baseline_me_scale = float(np.quantile(values.sum(1, dtype=np.float64)/bins, quantile))
    floor = max(DEVIANCE_SCALE_FLOOR, float(floor_fraction*baseline_me_scale))
    raw_bins = np.asarray(np.quantile(values, quantile, axis=0), dtype=np.float64)
    bin_scales = np.maximum(raw_bins, floor)
    normalized = np.empty(observed, dtype=np.float64)
    for first in range(0, observed, 4096):
        last = min(first+4096, observed)
        normalized[first:last] = (values[first:last]/bin_scales).sum(1)
    me_scale_raw = float(np.quantile(normalized, quantile))
    me_scale = max(me_scale_raw, DEVIANCE_SCALE_FLOOR)
    if (not np.isfinite(bin_scales).all() or not np.isfinite(me_scale)
            or not np.isfinite(baseline_me_scale) or not np.isfinite(floor)):
        raise ValueError('Good-Train deviance normalization produced nonfinite scales')
    return dict(bin_scales=bin_scales, me_scale=me_scale, me_scale_raw=me_scale_raw,
                me_floored=bool(me_scale_raw < DEVIANCE_SCALE_FLOOR),
                observed=int(observed), native_bins=int(bins), quantile=float(quantile),
                floor_fraction=float(floor_fraction), floor=floor,
                numerical_floor=DEVIANCE_SCALE_FLOOR, baseline_me_scale=baseline_me_scale,
                floored_bins=int((raw_bins < floor).sum()), storage_dtype='float32',
                reference='Frozen 03 Good Train deviance contributions',
                estimator='per_bin_deviance_quantile_then_me_quantile')
