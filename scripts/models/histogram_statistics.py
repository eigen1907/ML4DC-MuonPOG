"""Raw-count Input–Predict deviance, Pearson statistics and approximate p-values."""

import numpy as np
import torch


PROBABILITY_FLOOR = 1e-12


def normalize_probability(probability):
    """Apply the same positive probability floor in training and inference."""
    if isinstance(probability, torch.Tensor):
        q = probability.clamp_min(PROBABILITY_FLOOR)
        return q/q.sum(-1, keepdim=True)
    q = np.asarray(probability, dtype=np.float64)
    if (q.ndim < 1 or not q.shape[-1] or not np.isfinite(q).all()
            or (q < 0).any() or (q.sum(-1) <= 0).any()):
        raise ValueError('Expected finite nonnegative bin probabilities with positive totals')
    q = np.maximum(q, PROBABILITY_FLOOR)
    return q/q.sum(-1, keepdims=True)


def _counts_and_expected(counts, probability):
    q = normalize_probability(probability)
    if isinstance(q, torch.Tensor):
        n = torch.as_tensor(counts, dtype=q.dtype, device=q.device)
        if n.shape != q.shape:
            raise ValueError('Counts and probabilities must have matching native-bin shapes')
        total = n.sum(-1, keepdim=True)
    else:
        n = np.asarray(counts, dtype=np.float64)
        if (n.shape != q.shape or not np.isfinite(n).all() or (n < 0).any()):
            raise ValueError('Expected finite nonnegative native-bin counts matching the probabilities')
        total = n.sum(-1, keepdims=True)
    return n, total, total*q


def deviance_bins(counts, probability):
    """Per-bin 2[n log(n/mu) - n + mu], with mu=sum(counts)*probability.

    The linear terms sum to zero, giving multinomial deviance. Retaining
    them makes each bin's nonnegative contribution interpretable, including
    zero-count bins. A local series avoids cancellation near n=mu in float32.
    Empty histograms return zero; callers mask them as unobserved.
    """
    n, total, mu = _counts_and_expected(counts, probability)
    if isinstance(n, torch.Tensor):
        safe_mu = torch.where(total > 0, mu, torch.ones_like(mu))
        safe_n = torch.where(n > 0, n, torch.ones_like(n))
        delta = n-mu
        x = delta/safe_mu
        direct = n*torch.log(safe_n/safe_mu)-delta
        t = x.clamp(-.01, .01)
        series = mu*t.square()*(.5-t/6+t.square()/12-t.pow(3)/20+t.pow(4)/30)
        return 2*torch.where(x.abs() < .01, series, direct).clamp_min(0)
    safe_mu = np.where(total > 0, mu, 1.)
    safe_n = np.where(n > 0, n, 1.)
    delta = n-mu
    x = delta/safe_mu
    direct = n*np.log(safe_n/safe_mu)-delta
    t = np.clip(x, -.01, .01)
    series = mu*t*t*(.5-t/6+t*t/12-t**3/20+t**4/30)
    return 2*np.maximum(np.where(np.abs(x) < .01, series, direct), 0.)


def deviance(counts, probability):
    """Total Input–Predict count deviance; this is the shared loss/ME score."""
    return deviance_bins(counts, probability).sum(-1)


def weighted_deviance(counts, probability, bin_scales, me_scale):
    """Fixed bin- and ME-weighted deviance, shared by training and scoring.

    This weighted score is not a chi-square statistic. Raw count statistics
    remain available separately through count_statistics.
    """
    terms = deviance_bins(counts, probability)
    if isinstance(terms, torch.Tensor):
        bins = torch.as_tensor(bin_scales, dtype=terms.dtype, device=terms.device)
        me = torch.as_tensor(me_scale, dtype=terms.dtype, device=terms.device)
        if (bins.shape != (terms.shape[-1],) or me.ndim != 0
                or not torch.isfinite(bins).all() or not torch.isfinite(me)
                or (bins <= 0).any() or me <= 0):
            raise ValueError('Expected positive fixed native-bin scales and one positive ME scale')
    else:
        bins, me = np.asarray(bin_scales, dtype=float), np.asarray(me_scale, dtype=float)
        if (bins.shape != (terms.shape[-1],) or me.ndim != 0
                or not np.isfinite(bins).all() or not np.isfinite(me)
                or (bins <= 0).any() or me <= 0):
            raise ValueError('Expected positive fixed native-bin scales and one positive ME scale')
    return (terms/bins).sum(-1)/me


def count_statistics(counts, probability):
    """NumPy diagnostics, using asymptotic chi-square(B-1) only as a diagnostic.

    N is the sum of the tested native-bin contents, not ROOT GetEntries.
    These p-values assume a correct fixed prediction and independent counts;
    sparse bins and the learned, correlated ME prediction require validation.
    """
    from scipy.stats import chi2

    n, total, mu = _counts_and_expected(counts, probability)
    if isinstance(n, torch.Tensor):
        raise TypeError('Diagnostic p-values require NumPy counts and probabilities')
    observed = total[..., 0] > 0
    safe_mu = np.where(total > 0, mu, 1.)
    d = deviance(n, probability)
    x2 = ((n-mu)**2/safe_mu).sum(-1)
    df = n.shape[-1]-1
    p_d = chi2.sf(d, df) if df else np.ones_like(d)
    p_x2 = chi2.sf(x2, df) if df else np.ones_like(x2)
    values = dict(deviance=d, pearson=x2, p_deviance=p_d, p_pearson=p_x2,
                  expected_low_fraction=(mu < 5).mean(-1))
    return {**{key: np.where(observed, value, np.nan) for key, value in values.items()},
            'native_count': total[..., 0]}
