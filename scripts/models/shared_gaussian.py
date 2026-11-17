"""Train Gaussian references and fixed-weight fits that initialize free prefits."""

import hashlib
import json

import numpy as np

from scripts.models.gaussian import SCALE_FLOOR_BINS, fit_reference


def basis_fingerprint(basis):
    """Identify the Train reference by its contents, independent of experiment names."""
    reference = hashlib.sha256()
    for item in basis:
        for name in ('weights', 'means', 'scales', 'probabilities', 'edges'):
            values = np.ascontiguousarray(item[name], dtype='<f8')
            reference.update(f'{name}:{values.shape}'.encode())
            reference.update(memoryview(values))
        reference.update(json.dumps(dict(floor=float(item['floor']),
                                        periodic=bool(item['periodic'])), sort_keys=True).encode())
    return reference.hexdigest()


def fit_basis(histograms, axes, train_mask, components=16, iterations=80,
              reference_iterations=150, scale_floor_bins=SCALE_FLOOR_BINS):
    """Learn one fixed native-bin Gaussian basis per ME using training LS only.

    Every nonempty LS contributes one normalized histogram, so exposure does
    not determine its weight in the pooled shape. The pooled total equals the
    number of contributing LS. Locations/widths are learned once, with a common
    minimum sigma of scale_floor_bins * native bin width. Fixed-weight LS fits
    initialize the subsequent free-parameter prefits.
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
        reference = fit_reference(
            pooled, axis, components, iterations=iterations,
            reference_iterations=reference_iterations,
            scale_floor_bins=scale_floor_bins)
        reference['training_ls'] = observed
        basis.append(reference)
        print(f'Fixed reference {reference["stem"]}: {components} Gaussians', flush=True)
    return basis


def fit_weights(histograms, basis, iterations=60, batch_size=1024):
    """Fit only LS mixture fractions under fixed Gaussian shapes.

    EM maximizes the binned mixture likelihood without pseudo-events or a prior.
    Train reference weights initialize the fit only. Empty MEs have zero
    placeholder features and valid=False. No events are expanded.
    """
    if (not histograms or len(histograms) != len(basis)
            or type(iterations) is not int or iterations < 1
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError('Expected aligned MEs and positive iteration/batch sizes')
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
                weights = assigned / totals[:, None]
                weights /= weights.sum(1, keepdims=True)
            values[np.flatnonzero(observed) + start] = weights
        features.append(values)
        owners.extend([me] * len(reference))
        names.extend(f'{item["stem"]}__g{k}_weight' for k in range(len(reference)))
    return dict(features=np.concatenate(features, axis=1), valid=valid,
                feature_me=np.asarray(owners, dtype=np.int64), feature_names=names)
