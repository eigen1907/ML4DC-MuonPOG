"""Fit a fixed Gaussian reference to one pooled training histogram."""

import numpy as np
from scipy.special import ndtr
import torch


SCALE_FLOOR_BINS = 1 / np.sqrt(12)


def gaussian_bin_probabilities(weights, means, scales, edges, *, periodic=False):
    """Integrate normalized Gaussian mixtures over bins; wrap phi on a circle."""
    weights, means, scales = (np.asarray(x, dtype=np.float64) for x in (weights, means, scales))
    edges = np.asarray(edges, dtype=np.float64)
    if (weights.shape != means.shape or means.shape != scales.shape
            or weights.ndim < 1 or weights.shape[-1] < 1
            or edges.ndim != 1 or len(edges) < 2 or (np.diff(edges) < 0).any()
            or not np.isfinite(edges).all() or not edges[-1] > edges[0]
            or any(not np.isfinite(x).all() for x in (weights, means, scales))
            or (weights < 0).any() or (weights.sum(-1) <= 0).any() or (scales <= 0).any()):
        raise ValueError('Expected finite Gaussian parameters, positive scales and ordered bin edges')
    weights = weights / weights.sum(-1, keepdims=True)
    mu, sigma = means[..., None, :], scales[..., None, :]
    lower, upper = edges[:-1, None], edges[1:, None]
    probability = np.zeros(np.broadcast_shapes(mu.shape, lower.shape), dtype=np.float64)
    for image in (-2, -1, 0, 1, 2) if periodic else (0,):
        low, high = (lower - mu - image) / sigma, (upper - mu - image) / sigma
        right = low > 0
        probability += ndtr(np.where(right, -low, high)) - ndtr(np.where(right, -high, low))
    probability = np.maximum(probability, 0)
    probability /= np.maximum(probability.sum(-2, keepdims=True), np.finfo(float).tiny)
    probability = (probability * weights[..., None, :]).sum(-1)
    return probability / np.maximum(probability.sum(-1, keepdims=True), np.finfo(float).tiny)


def _axis(axis, bins, scale_floor_bins=SCALE_FLOOR_BINS):
    physical = np.asarray(axis['edges'] if 'edges' in axis else
                          np.linspace(axis['low'], axis['high'], bins + 1), dtype=float)
    if (physical.shape != (bins + 1,) or not np.isfinite(physical).all()
            or (np.diff(physical) <= 0).any() or not np.isfinite(scale_floor_bins) or scale_floor_bins <= 0):
        raise ValueError('Expected increasing finite bin edges and a positive minimum width')
    periodic = str(axis.get('stem', '')).endswith('__GlbMuon_Glb_phi')
    # Native phi axes have slight padding outside [-pi, pi].
    edges = (np.clip((physical + np.pi) / (2 * np.pi), 0, 1) if periodic else
             (physical - physical[0]) / (physical[-1] - physical[0]))
    native_width = float(np.median(np.diff(physical)))
    floor = scale_floor_bins * native_width / (2*np.pi if periodic else physical[-1]-physical[0])
    return ((edges[:-1] + edges[1:]) / 2).astype(np.float32), edges, floor, periodic


def _initial_fit(counts, centers, components, floor, periodic, iterations):
    """Initialize the reference by weighted bin-center EM; no event expansion."""
    counts = np.asarray(counts, dtype=np.float32)
    fractions = counts / counts.sum(dtype=np.float64)
    quantiles = (np.arange(components, dtype=np.float32) + .5) / components
    means = centers[np.searchsorted(fractions.cumsum(), quantiles).clip(max=len(centers)-1)]
    means = np.clip(means + floor * np.linspace(-.25, .25, components), floor/2, 1-floor/2)
    variance = (fractions * (centers - fractions @ centers) ** 2).sum()
    scales = np.full(components, np.clip(np.sqrt(variance/components), floor, .5))
    weights = np.full(components, 1/components)
    coordinates = centers[:, None, None] - (np.arange(-2, 3) if periodic else np.zeros(1))
    previous = -np.inf
    for iteration in range(iterations):
        delta = (coordinates - means[None, :, None]) / scales[None, :, None]
        # A numerical log(0) guard adds no pseudo-events to component masses.
        log_weights = np.log(np.maximum(weights, np.finfo(float).tiny))
        log_density = -.5 * delta**2 - np.log(scales)[None, :, None] + log_weights[None, :, None]
        maximum = log_density.max(axis=(1, 2), keepdims=True)
        density = np.exp(log_density - maximum)
        total = density.sum(axis=(1, 2), keepdims=True)
        likelihood = (fractions * (maximum[:, 0, 0] + np.log(total[:, 0, 0]))).sum()
        if iteration >= 2 and abs(likelihood - previous) < 1e-4:
            break
        previous = likelihood
        assigned = density / total * counts[:, None, None]
        mass = assigned.sum(axis=(0, 2))
        safe = np.maximum(mass, 1e-20)
        fitted_mean = (assigned * coordinates).sum(axis=(0, 2)) / safe
        fitted_variance = (assigned * (coordinates - fitted_mean[None, :, None])**2).sum(axis=(0, 2)) / safe
        means = np.where(mass < 1e-10, means, fitted_mean % 1 if periodic else np.clip(fitted_mean, 0, 1))
        scales = np.where(mass < 1e-10, scales, np.clip(np.sqrt(fitted_variance), floor, .5))
        weights = mass / mass.sum()
    return weights, means, scales


def _torch_components(means, scales, edges, periodic):
    """Differentiable native-bin probabilities for the one-time reference fit."""
    lower, upper = edges[:-1, None], edges[1:, None]
    probability = means.new_zeros((len(lower), len(means)))
    for image in (-2, -1, 0, 1, 2) if periodic else (0,):
        low, high = (lower-means-image)/scales, (upper-means-image)/scales
        right = low > 0
        probability = probability + .5 * (
            torch.erfc(-torch.where(right, -low, high)/np.sqrt(2))
            - torch.erfc(-torch.where(right, -high, low)/np.sqrt(2)))
    probability = probability.clamp_min(0)
    probability = probability / probability.sum(0, keepdim=True).clamp_min(1e-30)
    probability = probability.clamp_min(1e-12)
    return probability / probability.sum(0, keepdim=True)


def fit_reference(pooled, axis, components=16, *, iterations=80,
                  reference_iterations=150, scale_floor_bins=SCALE_FLOOR_BINS):
    """Fit locations, widths and weights once using native-bin likelihood."""
    pooled = np.asarray(pooled, dtype=np.float64)
    if (pooled.ndim != 1 or not np.isfinite(pooled).all() or (pooled < 0).any()
            or pooled.sum() <= 0 or type(components) is not int or components < 1):
        raise ValueError('Expected a nonempty nonnegative reference histogram and positive Gaussian count')
    centers, edges, floor, periodic = _axis(axis, len(pooled), scale_floor_bins)
    weights, means, scales = _initial_fit(pooled, centers, components, floor, periodic, iterations)
    logits = torch.tensor(np.log(np.maximum(weights, np.finfo(float).tiny)),
                          dtype=torch.float64, requires_grad=True)
    locations = torch.tensor(np.log(np.clip(means, 1e-8, 1-1e-8) / np.clip(1-means, 1e-8, 1)),
                             dtype=torch.float64, requires_grad=True)
    log_scales = torch.tensor(np.log(scales), dtype=torch.float64, requires_grad=True)
    edge_tensor = torch.tensor(edges, dtype=torch.float64)
    fractions = torch.tensor(pooled/pooled.sum(), dtype=torch.float64)
    optimizer = torch.optim.LBFGS([logits, locations, log_scales], lr=1.,
                                 max_iter=reference_iterations, tolerance_grad=1e-8,
                                 tolerance_change=1e-10, line_search_fn='strong_wolfe')

    def objective():
        optimizer.zero_grad()
        probability = _torch_components(locations.sigmoid(), log_scales.exp().clamp(floor, .5),
                                        edge_tensor, periodic) @ logits.softmax(0)
        loss = -(fractions * probability.clamp_min(1e-30).log()).sum()
        loss.backward()
        return loss

    optimizer.step(objective)
    weights = logits.softmax(0).detach().numpy()
    means = locations.sigmoid().detach().numpy()
    scales = log_scales.exp().clamp(floor, .5).detach().numpy()
    order = np.argsort(means, kind='stable')
    weights, means, scales = weights[order], means[order], scales[order]
    probabilities = gaussian_bin_probabilities(np.ones((components, 1)), means[:, None],
                                               scales[:, None], edges, periodic=periodic).T
    probabilities = np.maximum(probabilities, 1e-12)
    probabilities /= probabilities.sum(0, keepdims=True)
    return dict(weights=weights, means=means, scales=scales, probabilities=probabilities,
                edges=edges, periodic=periodic, floor=floor, stem=axis.get('stem', 'ME'))
