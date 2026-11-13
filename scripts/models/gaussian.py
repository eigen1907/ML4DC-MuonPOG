"""Small binned Gaussian-mixture fits, one fit per ME and LS.

EM treats bin centres as weighted observations; no counts are expanded into
individual samples. Reported likelihoods integrate each Gaussian over the
actual bin edges. This is a descriptive shape fit, not an anomaly score.
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


def parameters_from_features(features, axes, components, *, scale_floor_bins=SCALE_FLOOR_BINS):
    """Project decoded shape features back to valid Gaussian parameters.

    Negative predicted weights are clipped and renormalized; means use native
    support (circular atan2 for phi), and scales retain the original fitting floor.
    Observed N=0 masks must still be applied by the caller when scoring shapes.
    """
    features = np.asarray(features, dtype=np.float32)
    if (features.ndim != 2 or not np.isfinite(features).all() or type(components) is not int
            or components < 1 or not np.isfinite(scale_floor_bins) or scale_floor_bins <= 0):
        raise ValueError("Expected finite LS-by-feature values and positive mixture sizes/floors")
    result = np.empty((len(features), len(axes), components, 3), dtype=np.float32)
    start = 0
    for me, axis in enumerate(axes):
        if "edges" not in axis:
            raise ValueError("Native axis edges are required to decode fitted features")
        _, _, floor, periodic = _axis(axis, len(axis["edges"]) - 1, scale_floor_bins)
        stop = start + components * (4 if periodic else 3)
        if stop > features.shape[1]:
            raise ValueError("Feature count does not match the Gaussian representation")
        block = features[:, start:stop].reshape(len(features), components, -1)
        weights = np.maximum(block[..., 0], 1e-8)
        result[:, me, :, 0] = weights / weights.sum(-1, keepdims=True)
        result[:, me, :, 1] = (np.arctan2(block[..., 1], block[..., 2]) / (2 * np.pi) % 1
                               if periodic else np.clip(block[..., 1], 0, 1))
        result[:, me, :, 2] = np.clip(block[..., -1], floor, .5)
        start = stop
    if start != features.shape[1]:
        raise ValueError("Feature count does not match the Gaussian representation")
    return result


def probabilities_from_features(features, axes, components, *, scale_floor_bins=SCALE_FLOOR_BINS,
                                batch_size=512):
    """Decode AE outputs into native-bin probabilities without large CDF temporaries."""
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Expected a positive decoding batch size")
    parameters = parameters_from_features(features, axes, components, scale_floor_bins=scale_floor_bins)
    probabilities = []
    for me, axis in enumerate(axes):
        _, edges, _, periodic = _axis(axis, len(axis["edges"]) - 1, scale_floor_bins)
        probability = np.empty((len(features), len(edges) - 1), dtype=np.float32)
        for first in range(0, len(features), batch_size):
            last = min(len(features), first + batch_size)
            p = parameters[first:last, me]
            probability[first:last] = gaussian_bin_probabilities(
                p[..., 0], p[..., 1], p[..., 2], edges, periodic=periodic)
        probabilities.append(probability)
    return probabilities


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
    """Return Gaussian parameters and shape features for each LS/ME.

    parameters: (LS, ME, K, 3), ordered as weight, mean, scale, with components
    sorted by mean. Other MEs use their full native range, including signed eta.
    Phi uses true circular coordinates; its feature means are sine/cosine pairs.
    features: 3K columns per ordinary ME, 4K per phi ME. Empty MEs have zero
    features, valid=False and NaN fit diagnostics; their parameters are placeholders.
    Low-count fits remain weakly identified despite the bin-resolution scale floor and
    weight pseudocount. No training labels or other LS enter any individual fit.
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
    fit_nll, fit_deviance = (np.full(shape, np.nan) for _ in range(2))
    n_iterations = np.zeros(shape, dtype=np.int16)
    converged = np.zeros(shape, dtype=bool)
    features, names, feature_me, normalized_edges, periodic_axes = [], [], [], [], []
    for me, (histogram, axis) in enumerate(zip(arrays, axes)):
        centers, edges, floor, periodic = _axis(axis, histogram.shape[1], scale_floor_bins)
        normalized_edges.append(edges)
        periodic_axes.append(periodic)
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
            probability = gaussian_bin_probabilities(fitted[..., 0], fitted[..., 1], fitted[..., 2], edges, periodic=periodic)
            fractions = h[observed] / total[observed, None]
            log_probability = np.log(np.maximum(probability, 1e-300))
            fit_nll[rows, me] = -(fractions * log_probability).sum(1)
            empirical = np.log(np.maximum(fractions, 1e-300))
            fit_deviance[rows, me] = np.maximum(0, 2 * (fractions * (empirical - log_probability)).sum(1))
        values = parameters[:, me]
        pieces = [values[..., 0]]
        kinds = ["weight"]
        if periodic:
            pieces.extend([np.sin(2 * np.pi * values[..., 1]), np.cos(2 * np.pi * values[..., 1])])
            kinds.extend(["mean_sin", "mean_cos"])
        else:
            pieces.append(values[..., 1])
            kinds.append("mean")
        pieces.append(values[..., 2])
        kinds.append("scale")
        block = np.stack(pieces, axis=-1).reshape(size, -1)
        block[~valid[:, me]] = 0
        features.append(block)
        feature_me.extend([me] * block.shape[1])
        stem = axis.get("stem", f"me_{me}")
        names.extend(f"{stem}__g{component}_{kind}" for component in range(components) for kind in kinds)
    return dict(features=np.concatenate(features, axis=1), parameters=parameters,
                valid=valid, totals=totals, fit_nll=fit_nll, fit_deviance=fit_deviance,
                iterations=n_iterations, converged=converged, feature_names=names,
                feature_me=np.asarray(feature_me, dtype=np.int64),
                normalized_edges=normalized_edges, periodic=periodic_axes, components=components,
                scale_floor_bins=float(scale_floor_bins))
