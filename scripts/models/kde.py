"""Binned Gaussian KDE shapes with fixed or training-adaptive bandwidths.

Counts weight kernels at bin centres because the cache has no unbinned values.
This is a KDE representation for the LS autoencoder, not a standalone joint
likelihood model. Kernels are integrated over bins and normalized in range.
"""

import numpy as np

from scripts.models.gaussian import _axis, gaussian_bin_probabilities


def _adaptive_reference(histogram, train_rows, item, multipliers, bounds,
                        sample_limit, prior_strength, seed):
    """Training-only count-split selection of bounded Abramson bandwidths."""
    bins = histogram.shape[1]
    centers, edges = item['centers'], item['edges']
    widths = np.diff(edges)
    native_width = float(np.median(widths[widths > 0]))

    def kernels(raw, multiplier):
        pilot = item['probabilities'] @ raw
        density = np.maximum(pilot / np.maximum(widths, native_width * 1e-12),
                             np.finfo(float).tiny)
        log_geometric = float(raw @ np.log(density))
        factors = np.exp(np.clip(np.log(multiplier) + .5 * (log_geometric - np.log(density)),
                                 np.log(bounds[0]), np.log(bounds[1])))
        bandwidths = np.minimum(.5, item['bandwidth'] * factors)
        probability = gaussian_bin_probabilities(
            np.ones((bins, 1)), centers[:, None], bandwidths[:, None], edges,
            periodic=item['periodic']).T
        return probability, bandwidths

    # Selected rows are withheld entirely from the pilot/prior used for CV.
    # The final kernel uses every Good training LS only after selection.
    rng = np.random.default_rng(seed)
    wanted = min(sample_limit, item['training_ls'] // 2)
    candidates = rng.choice(train_rows, min(len(train_rows), 4 * wanted), replace=False)
    counts = histogram[candidates].astype(np.float64)
    if not np.equal(counts, np.floor(counts)).all():
        raise ValueError('Adaptive bandwidth count splitting requires integer counts')
    eligible = counts.sum(1) >= 2
    candidates, counts = candidates[eligible][:wanted], counts[eligible][:wanted]
    first = rng.binomial(counts.astype(np.int64), .5)
    second = counts - first
    valid = (first.sum(1) > 0) & (second.sum(1) > 0)
    candidates, counts, first, second = (x[valid] for x in (candidates, counts, first, second))
    scores, multiplier = [], 1.
    if len(candidates):
        held = (counts / counts.sum(1, keepdims=True)).sum(0)
        background = np.maximum(item['raw_reference'] * item['training_ls'] - held, 0)
        background /= background.sum()
        observed = second / second.sum(1, keepdims=True)
        mixture = ((first + prior_strength * background)
                   / (first.sum(1, keepdims=True) + prior_strength))
        for candidate in multipliers:
            probability, _ = kernels(background, candidate)
            predicted = mixture @ probability.T
            ce = float(-(observed * np.log(np.maximum(predicted, 1e-12))).sum(1).mean())
            scores.append(dict(multiplier=float(candidate), cross_entropy=ce))
        multiplier = min(scores, key=lambda row: row['cross_entropy'])['multiplier']
    probability, bandwidths = kernels(item['raw_reference'], multiplier)
    reference = probability @ item['raw_reference']
    reference /= reference.sum()
    return dict(probabilities=probability, reference=reference,
                adaptive=dict(method='train-count-split Abramson', multiplier=multiplier,
                              bounds=tuple(bounds), bandwidths=bandwidths,
                              bandwidth_bins_min=float(bandwidths.min() / native_width),
                              bandwidth_bins_median=float(np.median(bandwidths) / native_width),
                              bandwidth_bins_max=float(bandwidths.max() / native_width),
                              cv_rows=candidates.tolist(), cv_ls=int(len(candidates)),
                              pilot_ls=item['training_ls'] - len(candidates),
                              cv_scores=scores, seed=int(seed)))


def train_reference(histograms, axes, train_mask, bandwidth_bins=1., *, adaptive=False,
                    adaptive_multipliers=(.5, 1., 2.), adaptive_clip=(.5, 2.),
                    adaptive_cv_ls=256, adaptive_cv_prior=8., adaptive_seed=42):
    """Build fixed kernels and an equal-LS Good-training reference per ME.

    The baseline width is one native bin. Phi uses wrapped kernels.
    raw_reference contains bin fractions before smoothing, so prior shrinkage
    followed by KDE applies the smoothing exactly once.

    adaptive=True allows narrower kernels in dense pilot regions and broader
    kernels in sparse regions (bounded relative to the fixed bandwidth).
    Per-ME multipliers minimize held-out-count cross entropy within Good TRAIN.
    The CV rows are excluded from their selection pilot; final kernels then use
    all training LS. Half-bin adaptive widths are representation bandwidths,
    not a claim to recover unbinned information. The default fixed path is kept.
    """
    if not histograms or len(histograms) != len(axes):
        raise ValueError('Expected matching histogram and axis lists')
    if not np.isfinite(bandwidth_bins) or bandwidth_bins < 1:
        raise ValueError('KDE bandwidth must be at least one native bin')
    if adaptive:
        multipliers = np.asarray(adaptive_multipliers, dtype=float)
        bounds = np.asarray(adaptive_clip, dtype=float)
        if (multipliers.ndim != 1 or not len(multipliers) or not np.isfinite(multipliers).all()
                or (multipliers <= 0).any() or bounds.shape != (2,)
                or not np.isfinite(bounds).all() or not 0 < bounds[0] <= bounds[1]
                or type(adaptive_cv_ls) is not int or adaptive_cv_ls < 1
                or not np.isfinite(adaptive_cv_prior) or adaptive_cv_prior < 0
                or type(adaptive_seed) is not int or adaptive_seed < 0):
            raise ValueError('Expected positive bounded adaptive widths and a valid TRAIN CV selection')
    size = len(histograms[0])
    train = np.asarray(train_mask)
    if train.dtype != bool or train.shape != (size,) or not train.any():
        raise ValueError('Expected a nonempty boolean Good-training selection')
    references = []
    rows = np.flatnonzero(train)
    for histogram, axis in zip(histograms, axes):
        histogram = np.asarray(histogram)
        if histogram.ndim != 2 or len(histogram) != size or histogram.shape[1] < 1:
            raise ValueError('Expected aligned LS-by-bin histograms')
        bins = histogram.shape[1]
        centers, edges, _, periodic = _axis(axis, bins)
        widths = np.diff(edges)
        width = min(.5, float(np.median(widths[widths > 0])) * bandwidth_bins)
        # One Gaussian per row, not a B-component one-hot mixture (which needs B^3 temporaries).
        kernels = gaussian_bin_probabilities(
            np.ones((bins, 1)), centers[:, None], np.full((bins, 1), width),
            edges, periodic=periodic).T
        pooled, observed = np.zeros(bins, dtype=np.float64), 0
        for start in range(0, len(rows), 4096):
            counts = histogram[rows[start:start + 4096]].astype(np.float64)
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError('Expected finite nonnegative histogram counts')
            totals = counts.sum(1)
            valid = totals > 0
            pooled += (counts[valid] / totals[valid, None]).sum(0)
            observed += int(valid.sum())
        if not observed:
            raise ValueError('An ME has no observed Good training histogram')
        pooled /= observed
        reference = kernels @ pooled
        reference /= reference.sum()
        item = dict(probabilities=kernels, raw_reference=pooled,
                    reference=reference, centers=centers, edges=edges,
                    bandwidth=width, bandwidth_bins=float(bandwidth_bins),
                    periodic=periodic, training_ls=observed,
                    stem=axis.get('stem', f'me_{len(references)}'))
        if adaptive:
            item.update(_adaptive_reference(histogram, rows, item, multipliers, bounds,
                                            adaptive_cv_ls, adaptive_cv_prior, adaptive_seed))
        references.append(item)
    return references


def smooth(histograms, reference, prior_strength=8., batch_size=1024):
    """Return regularized KDE bin fractions as features, masking N=0 MEs.

    The total prior mass is in pseudo-observations, matching the shared-GMM
    convention. Setting it to zero gives the ordinary bin-centre weighted KDE.
    An empty histogram stays missing rather than becoming the normal reference.
    """
    if (not histograms or len(histograms) != len(reference)
            or not np.isfinite(prior_strength) or prior_strength < 0
            or type(batch_size) is not int or batch_size < 1):
        raise ValueError('Expected aligned MEs, nonnegative prior and positive batch size')
    size = len(histograms[0])
    features, names, owners = [], [], []
    valid = np.zeros((size, len(histograms)), dtype=bool)
    for me, (histogram, item) in enumerate(zip(histograms, reference)):
        histogram = np.asarray(histogram)
        kernels = np.asarray(item['probabilities'], dtype=np.float64)
        prior = np.asarray(item['raw_reference'], dtype=np.float64)
        if (histogram.ndim != 2 or len(histogram) != size
                or kernels.shape != (histogram.shape[1], histogram.shape[1])
                or prior.shape != (histogram.shape[1],)):
            raise ValueError('Histogram bins differ from the saved KDE reference')
        values = np.zeros(histogram.shape, dtype=np.float32)
        for start in range(0, size, batch_size):
            stop = min(size, start + batch_size)
            counts = histogram[start:stop].astype(np.float64)
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError('Expected finite nonnegative histogram counts')
            totals = counts.sum(1)
            observed = totals > 0
            valid[start:stop, me] = observed
            if not observed.any():
                continue
            fractions = ((counts[observed] + prior_strength * prior)
                         / (totals[observed, None] + prior_strength))
            values[np.flatnonzero(observed) + start] = fractions @ kernels.T
        features.append(values)
        owners.extend([me] * histogram.shape[1])
        names.extend(f'{item["stem"]}__kde_bin_{bin_number}'
                     for bin_number in range(histogram.shape[1]))
    return dict(features=np.concatenate(features, axis=1), valid=valid,
                feature_me=np.asarray(owners, dtype=np.int64), feature_names=names)


def raw_features(histograms):
    """Original normalized bin fractions as AE targets; no KDE or prior.

    Empty MEs have zero placeholders, with observation masks supplied by the
    KDE input preparation. One preallocated feature array avoids a full copy
    when joining the native-bin blocks.
    """
    if not histograms:
        raise ValueError('Expected a nonempty histogram list')
    arrays = [np.asarray(h) for h in histograms]
    size = len(arrays[0])
    if any(h.ndim != 2 or len(h) != size or h.shape[1] < 1 for h in arrays):
        raise ValueError('Expected aligned LS-by-bin histograms')
    values = np.zeros((size, sum(h.shape[1] for h in arrays)), dtype=np.float32)
    start = 0
    for histogram in arrays:
        stop = start + histogram.shape[1]
        for first in range(0, size, 4096):
            last = min(size, first + 4096)
            counts = histogram[first:last].astype(np.float64)
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError('Expected finite nonnegative histogram counts')
            totals = counts.sum(1, keepdims=True)
            np.divide(counts, totals, out=values[first:last, start:stop], where=totals > 0)
        start = stop
    return values


def decode(features, reference):
    """Project raw-histogram AE outputs to normalized native-bin probabilities.

    No kernel convolution is applied here: the decoder predicts the original
    unsmoothed fractions. An all-negative ME falls back to the raw training
    histogram reference. The smoothed reference is only an input diagnostic;
    N=0 masks must still be applied when scoring the original histogram.
    """
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or not np.isfinite(features).all():
        raise ValueError('Expected finite LS-by-KDE-feature values')
    probabilities, start = [], 0
    for item in reference:
        fallback = np.asarray(item['raw_reference'], dtype=np.float64)
        stop = start + len(fallback)
        if stop > features.shape[1]:
            raise ValueError('Feature count differs from the saved KDE reference')
        values = np.maximum(features[:, start:stop].astype(np.float64), 0)
        total = values.sum(1, keepdims=True)
        values = np.divide(values, total,
                           out=np.broadcast_to(fallback, values.shape).copy(), where=total > 0)
        probabilities.append(values.astype(np.float32))
        start = stop
    if start != features.shape[1]:
        raise ValueError('Feature count differs from the saved KDE reference')
    return probabilities
