"""Weighted binned Gaussian fitting and native-bin probabilities.

EM treats bin centres as weighted observations without expanding events.
The shared model uses these fits to learn fixed Gaussian shapes from pooled
Good training histograms; anomaly scoring lives in shared_gaussian.py.
"""

import numpy as np
from scipy.special import ndtr


SCALE_FLOOR_BINS = 1 / np.sqrt(12)


def gaussian_bin_probabilities(weights, means, scales, edges, *, periodic=False):
    """Integrate mixtures over normalized bins, including finite support.

    Arrays have shape (..., components); edges have shape (bins + 1,).
    Each nonperiodic Gaussian is truncated to the observed interval. For phi,
    means are modulo one and five wrapped images cover scales up to 0.5.
    """
    weights, means, scales = (np.asarray(x, dtype=np.float64) for x in (weights, means, scales))
    edges = np.asarray(edges, dtype=np.float64)
    if (weights.shape != means.shape or means.shape != scales.shape
            or weights.ndim < 1 or weights.shape[-1] < 1
            or edges.ndim != 1 or len(edges) < 2 or (np.diff(edges) < 0).any()
            or not np.isfinite(edges).all() or not edges[-1] > edges[0]
            or any(not np.isfinite(x).all() for x in (weights, means, scales))
            or (weights < 0).any() or (weights.sum(-1) <= 0).any() or (scales <= 0).any()):
        raise ValueError("Expected finite mixture parameters, positive scales and ordered bin edges")
    weights = weights / weights.sum(-1, keepdims=True)
    mu, sigma = means[..., None, :], scales[..., None, :]
    lower, upper = edges[:-1, None], edges[1:, None]
    probability = np.zeros(np.broadcast_shapes(mu.shape, lower.shape), dtype=np.float64)
    for image in (-2, -1, 0, 1, 2) if periodic else (0,):
        low = (lower - mu - image) / sigma
        high = (upper - mu - image) / sigma
        positive = low > 0
        # Survival-function subtraction avoids 1 - 1 cancellation in the right tail.
        probability += ndtr(np.where(positive, -low, high)) - ndtr(np.where(positive, -high, low))
    probability = np.maximum(probability, 0)
    probability /= np.maximum(probability.sum(-2, keepdims=True), np.finfo(float).tiny)
    probability = (probability * weights[..., None, :]).sum(-1)
    probability /= np.maximum(probability.sum(-1, keepdims=True), np.finfo(float).tiny)
    return probability


def _axis(axis, bins, scale_floor_bins=SCALE_FLOOR_BINS):
    physical = np.asarray(axis["edges"] if "edges" in axis else
                          np.linspace(axis["low"], axis["high"], bins + 1), dtype=float)
    if (physical.shape != (bins + 1,) or not np.isfinite(physical).all()
            or (np.diff(physical) <= 0).any()):
        raise ValueError("Expected one increasing finite edge array per histogram")
    periodic = str(axis.get("stem", "")).endswith("__GlbMuon_Glb_phi")
    if periodic:
        # Native phi bins have a small padding outside [-pi, pi].
        edges = np.clip((physical + np.pi) / (2 * np.pi), 0, 1)
        centers = (edges[:-1] + edges[1:]) / 2
    else:
        edges = (physical - physical[0]) / (physical[-1] - physical[0])
        centers = (edges[:-1] + edges[1:]) / 2
    widths = np.diff(edges)
    floor = min(.5, float(np.median(widths[widths > 0])) * scale_floor_bins)
    return centers.astype(np.float32), edges, floor, periodic


def _fit_batch(counts, centers, components, floor, periodic, iterations, tolerance, pseudocount):
    totals = counts.sum(1)
    normalized = counts / totals[:, None]
    cumulative = normalized.cumsum(1)
    quantiles = (np.arange(components, dtype=np.float32) + .5) / components
    indices = np.stack([(cumulative >= q).argmax(1) for q in quantiles], axis=1)
    means = centers[indices]
    # Break repeated-quantile ties deterministically; a bin is the resolution limit.
    means = np.clip(means + floor * np.linspace(-.25, .25, components), floor / 2, 1 - floor / 2)
    variance = (normalized * (centers[None, :] - (normalized @ centers)[:, None]) ** 2).sum(1)
    scales = np.broadcast_to(np.maximum(np.sqrt(variance) / np.sqrt(components), floor)[:, None], means.shape).copy()
    scales = np.clip(scales, floor, .5)
    weights = np.full(means.shape, 1 / components)
    means, scales, weights = (x.astype(np.float32) for x in (means, scales, weights))
    previous = np.full(len(counts), -np.inf)
    converged = np.zeros(len(counts), dtype=bool)
    used = np.zeros(len(counts), dtype=np.int16)
    shifts = np.arange(-2, 3, dtype=np.float32) if periodic else np.zeros(1, dtype=np.float32)
    coordinates = centers[None, :, None, None] - shifts[None, None, None, :]
    for iteration in range(iterations):
        rows = np.flatnonzero(~converged)
        if not len(rows):
            break
        mu, sigma, mixing = means[rows], scales[rows], weights[rows]
        delta = (coordinates - mu[:, None, :, None]) / sigma[:, None, :, None]
        log_density = -.5 * delta ** 2 - np.log(sigma[:, None, :, None]) + np.log(mixing[:, None, :, None])
        maximum = log_density.max(axis=(2, 3), keepdims=True)
        density = np.exp(log_density - maximum)
        denominator = density.sum(axis=(2, 3), keepdims=True)
        responsibility = density / denominator
        log_like = (normalized[rows] * (maximum[..., 0, 0] + np.log(denominator[..., 0, 0]))).sum(1)
        stable = (iteration >= 2) & (np.abs(log_like - previous[rows]) < tolerance)
        converged[rows[stable]] = True
        previous[rows] = log_like
        used[rows] = iteration + 1
        rows, responsibility = rows[~stable], responsibility[~stable]
        if not len(rows):
            continue
        assigned = responsibility * counts[rows, :, None, None]
        mass = assigned.sum(axis=(1, 3))
        safe_mass = np.maximum(mass, 1e-20)
        new_means = (assigned * coordinates).sum(axis=(1, 3)) / safe_mass
        new_variance = (assigned * (coordinates - new_means[:, None, :, None]) ** 2).sum(axis=(1, 3)) / safe_mass
        missing = mass < 1e-10
        means[rows] = np.where(missing, means[rows], new_means % 1 if periodic else np.clip(new_means, 0, 1))
        scales[rows] = np.where(missing, scales[rows], np.clip(np.sqrt(np.maximum(new_variance, 0)), floor, .5))
        weights[rows] = (mass + pseudocount) / (totals[rows, None] + components * pseudocount)
    order = np.argsort(means, axis=1, kind="stable")
    parameters = np.stack([np.take_along_axis(x, order, axis=1) for x in (weights, means, scales)], axis=-1)
    return parameters, used, converged


def fit_mixtures(histograms, axes, components=2, *, iterations=20, batch_size=512,
                 tolerance=1e-4, pseudocount=.5, scale_floor_bins=SCALE_FLOOR_BINS):
    """Fit weighted histograms and return parameters, masks and convergence data.

    Parameters have shape (histogram, ME, component, 3), storing weight,
    normalized location and width, sorted by location. Phi wraps on a circle.
    Empty histograms have valid=False and zero placeholder parameters.
    """
    if (type(components) is not int or components < 1 or type(iterations) is not int or iterations < 1
            or type(batch_size) is not int or batch_size < 1
            or not np.isfinite([tolerance, pseudocount, scale_floor_bins]).all()
            or tolerance < 0 or pseudocount < 0 or scale_floor_bins <= 0):
        raise ValueError("Expected positive integer fit sizes and nonnegative regularization")
    if len(histograms) != len(axes) or not histograms:
        raise ValueError("Expected matching nonempty histogram and axis lists")
    arrays = [np.asarray(h) for h in histograms]
    size = len(arrays[0])
    if any(h.ndim != 2 or len(h) != size or h.shape[1] == 0 for h in arrays):
        raise ValueError("Expected aligned LS-by-bin arrays")
    shape = (size, len(arrays))
    parameters = np.zeros((*shape, components, 3), dtype=np.float32)
    valid = np.zeros(shape, dtype=bool)
    totals = np.zeros(shape, dtype=np.float64)
    n_iterations = np.zeros(shape, dtype=np.int16)
    converged = np.zeros(shape, dtype=bool)
    for me, (histogram, axis) in enumerate(zip(arrays, axes)):
        centers, _, floor, periodic = _axis(axis, histogram.shape[1], scale_floor_bins)
        for start in range(0, size, batch_size):
            stop = min(size, start + batch_size)
            h = np.asarray(histogram[start:stop], dtype=np.float32)
            if not np.isfinite(h).all() or (h < 0).any():
                raise ValueError("Expected finite nonnegative histogram contents")
            total = h.sum(1, dtype=np.float64)
            totals[start:stop, me] = total
            observed = total > 0
            valid[start:stop, me] = observed
            if not observed.any():
                continue
            rows = np.flatnonzero(observed) + start
            fitted, used, done = _fit_batch(h[observed], centers, components, floor, periodic,
                                           iterations, tolerance, pseudocount)
            parameters[rows, me] = fitted
            n_iterations[rows, me], converged[rows, me] = used, done
    return dict(parameters=parameters, valid=valid, totals=totals,
                iterations=n_iterations, converged=converged)
