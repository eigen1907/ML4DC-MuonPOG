"""Count-corrected MMD scores and Good-monitor ME rank references.

These helpers leave the trained model and its default score unchanged. Fit the
ME ECDFs on Good monitoring LS, then use the separate Good calibration LS to
set the aggregated rank-score threshold. Neither reference fitting nor
threshold selection should use Test LS.
"""

import numpy as np

from scripts.models.gaussian import _axis


def fit_ecdf(terms, reference_indices):
    """Return sorted finite monitoring scores for each ME, ready for saving."""
    terms = np.asarray(terms, dtype=float)
    indices = np.asarray(reference_indices)
    if (terms.ndim != 2 or not terms.shape[1] or np.isinf(terms).any()
            or indices.ndim != 1):
        raise ValueError('Expected LS-by-ME scores and monitoring row indices')
    if indices.dtype == bool:
        if indices.shape != (len(terms),):
            raise ValueError('Monitoring mask must match the LS rows')
    elif (not np.issubdtype(indices.dtype, np.integer)
          or (indices < 0).any() or (indices >= len(terms)).any()):
        raise ValueError('Monitoring indices must select existing LS rows')
    reference = [np.sort(column[np.isfinite(column)]) for column in terms[indices].T]
    if any(not len(column) for column in reference):
        raise ValueError('Each ME needs observed Good monitoring scores')
    return reference


def ecdf_terms(terms, reference):
    """Normalize ME scales with -log((1 + reference scores >= x) / (n + 1)).

    Missing MEs remain missing. This finite rank score is not a calibrated
    p-value: the smallest tail fraction is 1/(n+1), so scores above the largest
    monitoring value tie at log(n+1). No unsupported tail extrapolation is used.
    """
    terms = np.asarray(terms, dtype=float)
    if terms.ndim != 2 or terms.shape[1] != len(reference) or np.isinf(terms).any():
        raise ValueError('Expected one ECDF reference per ME score')
    normalized = np.full(terms.shape, np.nan)
    for me, column in enumerate(reference):
        column = np.asarray(column, dtype=float)
        if (column.ndim != 1 or not len(column) or not np.isfinite(column).all()
                or (np.diff(column) < 0).any()):
            raise ValueError('Expected sorted finite monitoring scores')
        valid = np.isfinite(terms[:, me])
        rank = np.searchsorted(column, terms[valid, me], side='left')
        normalized[valid, me] = -np.log((len(column) + 1 - rank) / (len(column) + 1))
    return normalized


def top_mean_score(terms, top_n=None):
    """Average all observed MEs, or the largest top_n observed ME scores.

    None preserves the original all-ME mean. Missing MEs are excluded; observed
    zeros count normally. Fewer than top_n observations use all available MEs,
    and an LS without observed MEs remains NaN.
    """
    terms = np.asarray(terms, dtype=float)
    if terms.ndim != 2 or not terms.shape[1] or np.isinf(terms).any():
        raise ValueError('Expected finite or missing LS-by-ME scores')
    valid = np.isfinite(terms)
    count = valid.sum(1)
    if top_n is None:
        total = np.where(valid, terms, 0).sum(1)
    else:
        if (isinstance(top_n, (bool, np.bool_))
                or not isinstance(top_n, (int, np.integer))
                or not 1 <= top_n <= terms.shape[1]):
            raise ValueError('top_n must be an integer from 1 to the number of MEs')
        selected = np.sort(np.where(valid, terms, -np.inf), axis=1)[:, -int(top_n):]
        total = np.where(np.isfinite(selected), selected, 0).sum(1)
        count = np.minimum(count, top_n)
    return np.divide(total, count, out=np.full(len(terms), np.nan), where=count > 0)


def mmd_kernels(axes, bin_sizes, bandwidth_bins=(1., 4., 16.)):
    """Precompute equal-weight multiscale Gaussian kernels on native bin centres.

    Bandwidths are in native bins. Phi uses the positive-definite circular
    Gaussian (chord distance), making the ends adjacent. Other axes use their
    normalized physical bin centres. Reuse these matrices across LS batches.
    """
    widths = np.asarray(bandwidth_bins, dtype=float)
    if (not axes or len(axes) != len(bin_sizes) or widths.ndim != 1
            or not len(widths) or not np.isfinite(widths).all() or (widths <= 0).any()):
        raise ValueError('Expected matching ME axes/bin sizes and positive bandwidths')
    kernels = []
    for axis, bins in zip(axes, bin_sizes):
        centers, edges, _, periodic = _axis(axis, bins)
        centers = centers.astype(float)
        distance = centers[:, None] - centers[None, :]
        if periodic:
            distance = np.sin(np.pi * distance) / np.pi
        positive = np.diff(edges)
        bandwidth = np.median(positive[positive > 0]) * widths
        kernel = np.exp(-.5 * (distance[..., None] / bandwidth) ** 2).mean(2)
        kernels.append(kernel)
    return kernels


def mmd_terms(histograms, probabilities, kernels, *, count_corrected=True, batch_size=1024):
    """Compare raw LS fractions and predicted probabilities, retaining N=0 masks.

    Raw mode returns T=(f-p)'K(f-p). Count-corrected mode returns the positive
    excess above E[T|N,p], divided by its fixed-p multinomial null std. Its
    moments are exact for independent integer counts, including N=1, rather
    than a large-N Gaussian approximation. Weighted/noninteger bins are rejected.

    This is a diagnostic, NOT a significance or a valid p-value: predictions
    depend on the observed LS, objects can share tracks/events, and the model
    has uncertainty. Calibrate the final score on held-out Good LS as usual.
    The native-bin approximation does not resolve structure within a bin.
    """
    if (not histograms or len(histograms) != len(probabilities)
            or len(histograms) != len(kernels) or type(batch_size) is not int or batch_size < 1):
        raise ValueError('Expected observations, predictions and a kernel per ME')
    size = len(histograms[0])
    terms = np.full((size, len(histograms)), np.nan)
    for me, (histogram, predicted, kernel) in enumerate(zip(histograms, probabilities, kernels)):
        histogram, predicted, kernel = map(np.asarray, (histogram, predicted, kernel))
        if (histogram.ndim != 2 or len(histogram) != size or predicted.shape != histogram.shape
                or kernel.shape != (histogram.shape[1], histogram.shape[1])
                or not np.isfinite(kernel).all() or not np.allclose(kernel, kernel.T)):
            raise ValueError('Expected aligned bins and a finite symmetric kernel')
        diagonal, squared = np.diag(kernel), kernel ** 2
        for start in range(0, size, batch_size):
            stop = min(size, start + batch_size)
            counts = histogram[start:stop].astype(float)
            p = predicted[start:stop].astype(float)
            if (not np.isfinite(counts).all() or (counts < 0).any()
                    or (np.abs(counts - np.rint(counts)) > 1e-5).any()
                    or not np.isfinite(p).all() or (p < 0).any()
                    or not np.allclose(p.sum(1), 1, rtol=1e-5, atol=1e-7)):
                raise ValueError('Expected nonnegative integer bins and normalized predictions')
            n = counts.sum(1)
            valid = n > 0
            if not valid.any():
                continue
            n, p, counts = n[valid], p[valid], counts[valid]
            delta = counts / n[:, None] - p
            raw = np.maximum(np.einsum('ij,ij->i', delta @ kernel, delta), 0)
            if count_corrected:
                kp = p @ kernel
                pkp = np.einsum('ij,ij->i', p, kp)
                a = np.maximum(p @ diagonal - pkp, 0)
                # a=E||U||², b=E||U||⁴, c=tr(Cov(U)²), U=phi(X)-E phi(X).
                b = (p * (diagonal - 2 * kp + pkp[:, None]) ** 2).sum(1)
                c = np.maximum(np.einsum('ij,ij->i', p @ squared, p)
                               - 2 * (p * kp ** 2).sum(1) + pkp ** 2, 0)
                variance = (np.maximum(b - a ** 2, 0) + 2 * (n - 1) * c) / n ** 3
                expected = a / n
                excess = np.maximum(raw - expected, 0)
                tolerance = 64 * np.finfo(float).eps * np.maximum(raw, expected)
                raw = np.where(excess > tolerance, excess / np.sqrt(np.maximum(variance, 1e-30)), 0)
            terms[np.flatnonzero(valid) + start, me] = raw
    return terms
