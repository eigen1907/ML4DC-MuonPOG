"""Compare each raw ME with a fixed Good Train Gaussian reference; no AE."""

import numpy as np

from scripts.models.config import REFERENCE_SCORE


NAME = REFERENCE_SCORE


def reference_terms(dataset, basis):
    """Native-bin fraction MSE to frozen Train shapes AND Train mixture weights.

    This comparison never fits mixture weights to the evaluated observation.
    Empty native histograms have no shape and retain a missing score.
    """
    histograms = dataset['histograms']
    if not histograms or len(histograms) != len(basis):
        raise ValueError('Expected one frozen reference per ME')
    size = len(histograms[0])
    terms = np.full((size, len(basis)), np.nan)
    for me, (histogram, item) in enumerate(zip(histograms, basis)):
        reference = np.asarray(item['probabilities']) @ np.asarray(item['weights'])
        if (histogram.ndim != 2 or len(histogram) != size
                or reference.shape != (histogram.shape[1],)
                or not np.isfinite(reference).all() or (reference < 0).any()
                or not np.isclose(reference.sum(), 1)):
            raise ValueError('Frozen reference and native histogram bins differ')
        for first in range(0, size, 4096):
            counts = histogram[first:first + 4096].astype(np.float64)
            if not np.isfinite(counts).all() or (counts < 0).any():
                raise ValueError('Expected finite nonnegative histogram counts')
            totals = counts.sum(1)
            observed = totals > 0
            values = counts[observed] / totals[observed, None]
            terms[first + np.flatnonzero(observed), me] = ((values-reference)**2).mean(1)
    return terms
