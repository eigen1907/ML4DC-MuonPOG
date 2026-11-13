"""Shared Gaussian shapes from Good training LS; only mixture weights vary by LS."""

import numpy as np

from scripts.models.gaussian import (_axis, SCALE_FLOOR_BINS, fit_mixtures,
                                    gaussian_bin_probabilities)


def fit_basis(histograms, axes, train_mask, components=8, iterations=80,
              scale_floor_bins=SCALE_FLOOR_BINS):
    """Learn one fixed native-bin Gaussian basis per ME using training LS only.

    Every nonempty LS contributes one normalized histogram, so exposure does
    not determine its weight in the pooled shape. The pooled total equals the
    number of contributing LS. Gaussian locations/widths are never refitted at
    inference. Phi uses the same wrapped density as the original GMM helper.
    """
    if not histograms or len(histograms) != len(axes):
        raise ValueError('Expected one histogram array and axis per ME')
    size = len(histograms[0])
    train = np.asarray(train_mask)
    if train.dtype != bool or train.shape != (size,) or not train.any():
        raise ValueError('Expected nonempty boolean training selection')
    basis = []
    for histogram, axis in zip(histograms, axes):
        histogram = np.asarray(histogram)
        if histogram.ndim != 2 or len(histogram) != size:
            raise ValueError('Expected aligned LS-by-bin histograms')
        pooled = np.zeros(histogram.shape[1], dtype=np.float64)
        observed = 0
        rows = np.flatnonzero(train)
        for start in range(0, len(rows), 1024):
            counts = histogram[rows[start:start + 1024]].astype(np.float64)
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError('Expected finite nonnegative histogram contents')
            totals = counts.sum(1)
            valid = totals > 0
            pooled += (counts[valid] / totals[valid, None]).sum(0)
            observed += int(valid.sum())
        if not observed:
            raise ValueError('An ME has no observed Good training histogram')
        fitted = fit_mixtures([pooled[None, :]], [axis], components,
                              iterations=iterations, scale_floor_bins=scale_floor_bins)
        weights, means, scales = fitted['parameters'][0, 0].T.astype(np.float64)
        _, edges, _, periodic = _axis(axis, histogram.shape[1], scale_floor_bins)
        probabilities = gaussian_bin_probabilities(
            np.eye(components), np.broadcast_to(means, (components, components)),
            np.broadcast_to(scales, (components, components)), edges,
            periodic=periodic).T
        probabilities = np.maximum(probabilities, 1e-12)
        probabilities /= probabilities.sum(0, keepdims=True)
        weights /= weights.sum()
        basis.append(dict(weights=weights, means=means, scales=scales,
                          probabilities=probabilities, edges=edges, periodic=periodic,
                          stem=axis.get('stem', f'me_{len(basis)}'), training_ls=observed))
    return basis


def fit_weights(histograms, basis, prior_strength=8., iterations=30, batch_size=1024):
    """Fit only LS mixture fractions under fixed Gaussian shapes.

    prior_strength is the TOTAL pseudo-event mass distributed according to the
    training reference weights. EM maximizes a binned mixture likelihood plus
    this Dirichlet MAP penalty. Empty MEs have zero placeholder features and
    valid=False; they are not normal observations. No events are expanded.
    """
    if (not histograms or len(histograms) != len(basis)
            or not np.isfinite(prior_strength) or prior_strength < 0
            or type(iterations) is not int or iterations < 1
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError('Expected aligned MEs and nonnegative weight regularization')
    size = len(histograms[0])
    features, names, owners = [], [], []
    valid = np.zeros((size, len(basis)), dtype=bool)
    for me, (histogram, item) in enumerate(zip(histograms, basis)):
        histogram = np.asarray(histogram)
        probability = np.asarray(item['probabilities'], dtype=np.float64)
        reference = np.asarray(item['weights'], dtype=np.float64)
        if (histogram.ndim != 2 or len(histogram) != size
                or probability.shape != (histogram.shape[1], len(reference))):
            raise ValueError('Histogram bins differ from the trained basis')
        values = np.zeros((size, len(reference)), dtype=np.float32)
        for start in range(0, size, batch_size):
            stop = min(size, start + batch_size)
            counts = histogram[start:stop].astype(np.float64)
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError('Expected finite nonnegative histogram contents')
            totals = counts.sum(1)
            observed = totals > 0
            valid[start:stop, me] = observed
            if not observed.any():
                continue
            counts, totals = counts[observed], totals[observed]
            weights = np.broadcast_to(reference, (len(counts), len(reference))).copy()
            for _ in range(iterations):
                predicted = weights @ probability.T
                assigned = weights * ((counts / predicted) @ probability)
                weights = (assigned + prior_strength * reference) / (totals[:, None] + prior_strength)
                weights /= weights.sum(1, keepdims=True)
            values[np.flatnonzero(observed) + start] = weights
        features.append(values)
        owners.extend([me] * len(reference))
        names.extend(f'{item["stem"]}__g{k}_weight' for k in range(len(reference)))
    return dict(features=np.concatenate(features, axis=1), valid=valid,
                feature_me=np.asarray(owners, dtype=np.int64), feature_names=names)


def decode_weights(features, basis):
    """Project decoded fractions to a simplex and return native-bin probabilities."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError('Expected finite LS-by-weight feature values')
    probabilities, start = [], 0
    for item in basis:
        reference = np.asarray(item['weights'], dtype=np.float64)
        stop = start + len(reference)
        if stop > features.shape[1]:
            raise ValueError('Feature count differs from the trained basis')
        weights = np.maximum(features[:, start:stop].astype(np.float64), 0)
        total = weights.sum(1, keepdims=True)
        weights = np.divide(weights, total, out=np.broadcast_to(reference, weights.shape).copy(),
                            where=total > 0)
        probabilities.append((weights @ item['probabilities'].T).astype(np.float32))
        start = stop
    if start != features.shape[1]:
        raise ValueError('Feature count differs from the trained basis')
    return probabilities


def training_variance(histograms, train_mask):
    """Empirical bin-fraction variance across nonempty Good training LS.

    Equal LS weighting includes both normal shape variation and counting noise.
    It is deliberately a conservative reference, not an intrinsic variance
    estimate with sampling fluctuations removed.
    """
    if not histograms:
        raise ValueError('Expected nonempty histogram list')
    size = len(histograms[0])
    train = np.asarray(train_mask)
    if train.dtype != bool or train.shape != (size,) or not train.any():
        raise ValueError('Expected nonempty boolean training selection')
    rows, variances = np.flatnonzero(train), []
    for histogram in histograms:
        histogram = np.asarray(histogram)
        if histogram.ndim != 2 or len(histogram) != size:
            raise ValueError('Expected aligned LS-by-bin histograms')
        summed, squared = np.zeros(histogram.shape[1]), np.zeros(histogram.shape[1])
        observed = 0
        for start in range(0, len(rows), 4096):
            counts = histogram[rows[start:start + 4096]].astype(np.float64)
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError('Expected finite nonnegative histogram contents')
            totals = counts.sum(1)
            valid = totals > 0
            fractions = counts[valid] / totals[valid, None]
            summed += fractions.sum(0)
            squared += (fractions * fractions).sum(0)
            observed += int(valid.sum())
        if not observed:
            raise ValueError('An ME has no observed Good training histogram')
        variances.append(np.maximum(squared / observed - (summed / observed) ** 2, 0))
    return variances


def histogram_terms(histograms, probabilities, variance, probability_floor):
    """Score original bins, including their count-dependent sampling variance.

    Per-ME score averages (fraction-p)^2 divided by
    v_train + max(p*(1-p), q*(1-q))/N. q is one training-only finite-exposure
    floor per ME, e.g. .5/(total_train_counts+.5*bins).
    Empty MEs are unscored. This diagnostic has an empirical Good-calibrated
    threshold; it is not a chi-square statistic or calibrated probability.
    """
    if not histograms or len(histograms) != len(probabilities) or len(histograms) != len(variance):
        raise ValueError('Expected observations, predictions and variance for every ME')
    probability_floor = np.asarray(probability_floor, dtype=np.float64)
    if (probability_floor.shape != (len(histograms),) or not np.isfinite(probability_floor).all()
            or (probability_floor <= 0).any() or (probability_floor > .5).any()):
        raise ValueError('Expected one positive training probability floor per ME, at most 0.5')
    size = len(histograms[0])
    terms = np.full((size, len(histograms)), np.nan)
    for me, (histogram, predicted, reference) in enumerate(zip(histograms, probabilities, variance)):
        histogram, predicted, reference = map(np.asarray, (histogram, predicted, reference))
        if (histogram.ndim != 2 or len(histogram) != size or predicted.shape != histogram.shape
                or reference.shape != (histogram.shape[1],) or not np.isfinite(reference).all()
                or (reference < 0).any()):
            raise ValueError('Expected aligned bins and finite nonnegative training variance')
        for start in range(0, size, 4096):
            stop = min(size, start + 4096)
            counts, p = histogram[start:stop].astype(np.float64), predicted[start:stop].astype(np.float64)
            if (not np.isfinite(counts).all() or (counts < 0).any()
                    or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
                    or not np.allclose(p.sum(1), 1, rtol=1e-5, atol=1e-7)):
                raise ValueError('Expected finite counts and normalized predicted probabilities')
            totals = counts.sum(1)
            valid = totals > 0
            p = p[valid]
            fractions = counts[valid] / totals[valid, None]
            q = probability_floor[me]
            sampling = np.maximum(p * (1 - p), q * (1 - q)) / totals[valid, None]
            denominator = np.maximum(reference + sampling, 1e-30)
            terms[np.flatnonzero(valid) + start, me] = ((fractions - p) ** 2 / denominator).mean(1)
    return terms
