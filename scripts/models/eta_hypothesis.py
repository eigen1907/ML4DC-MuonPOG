"""Exploratory eta acceptance-loss likelihoods, fitted on Good TRAIN LS only.

Two predeclared regions use native-bin centres: |eta| < 0.9 and |eta| > 1.2.
Each LS supplies a regional count k and in-range total N. A beta-binomial
normal reference accommodates Good shape variation beyond binomial noise.
The alternative mean reduces regional efficiency and renormalizes the
histogram, keeping a working beta-binomial dispersion. The returned profile
likelihood ratio is an anomaly score, NOT a
chi-square significance, p-value, or probability of a detector defect.
"""

import numpy as np
from scipy.special import betaln, xlogy


REGIONS = ('central', 'forward')
LOSSES = (0., .25, .5, .75, .95, .999)


def _regions(edges):
    edges = np.asarray(edges, dtype=float)
    if (edges.ndim != 1 or len(edges) < 3 or not np.isfinite(edges).all()
            or (np.diff(edges) <= 0).any()):
        raise ValueError('Expected increasing finite native eta bin edges')
    centres = (edges[:-1] + edges[1:]) / 2
    masks = np.column_stack((np.abs(centres) < .9, np.abs(centres) > 1.2))
    if not masks.any(0).all() or masks.all(0).any():
        raise ValueError('Eta axis must include each region and its complement')
    return masks


def _counts(counts, bins):
    counts = np.asarray(counts)
    if (counts.ndim != 2 or counts.shape[1] != bins
            or not np.isfinite(counts).all() or (counts < 0).any()
            or (not np.issubdtype(counts.dtype, np.integer)
                and (counts != np.floor(counts)).any())):
        raise ValueError('Expected LS-by-bin nonnegative integer eta counts')
    return counts


def _fit_parameters(regional, n):
    fractions = regional / n[:, None]
    inverse_n = float(np.mean(1 / n))
    parameters = []
    for values in fractions.T:
        p = float(np.clip(values.mean(), 1e-6, 1 - 1e-6))
        variance = float(values.var(ddof=1))
        scale = p * (1 - p)
        intrinsic = ((variance / scale - inverse_n) / (1 - inverse_n)
                     if inverse_n < 1 - 1e-12 else 0.)
        intrinsic = float(np.clip(intrinsic, 0., .95))
        reference_variance = variance / len(n)
        predictive = float(np.clip(intrinsic + reference_variance / scale, 0., .95))
        parameters.append(dict(probability=p, overdispersion=intrinsic,
                               reference_variance=reference_variance,
                               predictive_overdispersion=predictive))
    return dict(parameters=parameters, training_ls=len(n),
                training_entries=float(n.sum()), mean_inverse_entries=inverse_n)


def _pileup(pileup, size):
    pileup = np.asarray(pileup, dtype=float)
    if pileup.shape != (size,) or np.isinf(pileup).any() or (pileup < 0).any():
        raise ValueError('Expected nonnegative or missing pile-up per LS')
    return pileup


def fit_reference(counts, edges, train_mask, pileup=None):
    """Fit equal-LS regional means and finite-N-corrected overdispersion.

    train_mask must select Good TRAIN observations, never monitoring,
    calibration or Test. No LS count cut is imposed; N=0 supplies no shape.
    Under a beta-binomial model, Var(k/N) = p(1-p)[rho+(1-rho)E(1/N)].
    Solving this moment identity separates observed sampling variation from
    intrinsic Good variation. The small LS-mean estimation variance is also
    included conservatively in the predictive rho. It assumes independent
    LS and does not model temporal/run correlations; calibration stays needed.
    Optional pile-up uses at most four fixed TRAIN-quartile strata, with
    separate normal moments. Missing PU is excluded; a sparse stratum falls
    back to the global TRAIN reference. No Test/monitor/calibration fitting.
    """
    masks = _regions(edges)
    counts = _counts(counts, len(masks))
    train_mask = np.asarray(train_mask)
    if train_mask.dtype != bool or train_mask.shape != (len(counts),):
        raise ValueError('Expected one boolean Good TRAIN flag per LS')
    total = counts.sum(1, dtype=np.float64)
    selected = train_mask & (total > 0)
    if pileup is not None:
        pileup = _pileup(pileup, len(counts))
        selected &= np.isfinite(pileup)
    if selected.sum() < 2:
        raise ValueError('Reference requires at least two observed Good TRAIN LS')
    n = total[selected]
    regional = np.column_stack([counts[selected][:, mask].sum(1, dtype=float)
                                for mask in masks.T])
    reference = dict(edges=np.asarray(edges, dtype=float).tolist(), regions=list(REGIONS),
                     losses=list(LOSSES), **_fit_parameters(regional, n))
    if pileup is not None:
        pu = pileup[selected]
        boundaries = np.unique(np.quantile(pu, [.25, .5, .75]))
        boundaries = boundaries[(boundaries > pu.min()) & (boundaries < pu.max())]
        groups = np.searchsorted(boundaries, pu, side='right')
        strata = []
        for group in range(len(boundaries) + 1):
            rows = groups == group
            fitted = (_fit_parameters(regional[rows], n[rows]) if rows.sum() >= 2
                      else dict(parameters=reference['parameters'], training_ls=int(rows.sum()),
                                fallback_global=True))
            strata.append(fitted)
        reference.update(condition='pileup', pileup_boundaries=boundaries.tolist(), strata=strata)
    return reference


def _log_likelihood(k, n, probability, rho):
    """Omit the binomial coefficient, which cancels in every LR."""
    if rho < 1e-6:
        return xlogy(k, probability) + xlogy(n - k, 1 - probability)
    precision = 1 / rho - 1
    alpha, beta = probability * precision, (1 - probability) * precision
    return betaln(k + alpha, n - k + beta) - betaln(alpha, beta)


def score(counts, reference, pileup=None):
    """Return max regional 2*log(L_loss/L_normal), with interpretable terms.

    loss=0 is always available. loss=a implies q=(1-a)p/(1-a*p), the
    regional mean fraction after relative regional efficiency falls by a.
    Holding rho fixed is a working alternative, not the exact pushforward of
    a random beta-distributed probability through that transformation. Only
    deficits (observed k/N < p) can score. Central/forward share observations:
    take their maximum and calibrate it on held-out Good LS rather than sum
    them as independent evidence. N=0 remains NaN/unscored, never normal.
    """
    masks = _regions(reference['edges'])
    counts = _counts(counts, len(masks))
    if reference.get('regions') != list(REGIONS) or len(reference['parameters']) != 2:
        raise ValueError('Expected central and forward eta reference parameters')
    losses = np.asarray(reference['losses'], dtype=float)
    if (losses.ndim != 1 or not len(losses) or losses[0] != 0
            or not np.isfinite(losses).all() or (losses < 0).any() or (losses >= 1).any()):
        raise ValueError('Expected efficiency losses starting at zero and below one')
    total = counts.sum(1, dtype=np.float64)
    valid = total > 0
    groups = np.zeros(len(counts), dtype=int)
    parameter_sets = [reference['parameters']]
    if reference.get('condition') == 'pileup':
        if pileup is None:
            raise ValueError('This normal reference requires pile-up at inference')
        pileup = _pileup(pileup, len(counts))
        valid &= np.isfinite(pileup)
        boundaries = np.asarray(reference['pileup_boundaries'], dtype=float)
        if (not np.isfinite(boundaries).all() or (np.diff(boundaries) <= 0).any()
                or len(reference['strata']) != len(boundaries) + 1):
            raise ValueError('Invalid pile-up strata')
        groups = np.searchsorted(boundaries, pileup, side='right')
        parameter_sets = [stratum['parameters'] for stratum in reference['strata']]
    observed = np.full((len(counts), 2), np.nan)
    terms, fitted, severity, normal = [observed.copy() for _ in range(4)]
    for group, parameters in enumerate(parameter_sets):
        rows = valid & (groups == group)
        n = total[rows, None]
        selected_counts = counts[rows]
        for region, (mask, params) in enumerate(zip(masks.T, parameters)):
            p = params['probability']
            rho = params['predictive_overdispersion']
            if not np.isfinite([p, rho]).all() or not 0 < p < 1 or not 0 <= rho < 1:
                raise ValueError('Invalid beta-binomial normal reference')
            k = selected_counts[:, mask].sum(1, dtype=float)[:, None]
            observed[rows, region] = (k / n).ravel()
            normal[rows, region] = p
            q = (1 - losses) * p / (1 - losses * p)
            likelihood = _log_likelihood(k, n, q[None, :], rho)
            best = np.argmax(likelihood, axis=1)
            deficit = (k < p * n).ravel()
            best = np.where(deficit, best, 0)
            gain = likelihood[np.arange(len(best)), best] - likelihood[:, 0]
            terms[rows, region] = 2 * np.maximum(gain, 0)
            fitted[rows, region], severity[rows, region] = q[best], losses[best]
    aggregate = np.full(len(counts), np.nan)
    aggregate[valid] = terms[valid].max(1)
    return dict(score=aggregate, valid=valid, region_scores=terms,
                observed_fraction=observed, normal_probability=normal, fitted_probability=fitted,
                efficiency_loss=severity, entries=total)
