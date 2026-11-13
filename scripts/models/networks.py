"""Small pile-up-conditioned networks and Gaussian histogram likelihoods."""

import math

import torch
from torch import nn


class ConditionalAE(nn.Module):
    """Encode ME features with conditions; reconstruct only the ME features."""

    def __init__(self, input_dim, hidden=(32, 8), condition_dim=1):
        super().__init__()
        widths = tuple(int(width) for width in hidden)
        if input_dim < 1 or not widths or min(widths) < 1 or condition_dim < 0:
            raise ValueError('Expected positive feature/hidden sizes and nonnegative condition size')
        self.input_dim, self.condition_dim = int(input_dim), int(condition_dim)
        encoder, previous = [], self.input_dim + self.condition_dim
        for index, width in enumerate(widths):
            encoder.append(nn.Linear(previous, width))
            if index < len(widths) - 1:
                encoder.append(nn.SiLU())
            previous = width
        decoder, previous = [], widths[-1] + self.condition_dim
        for width in reversed(widths[:-1]):
            decoder.extend((nn.Linear(previous, width), nn.SiLU()))
            previous = width
        decoder.append(nn.Linear(previous, self.input_dim))
        self.encoder, self.decoder = nn.Sequential(*encoder), nn.Sequential(*decoder)

    def forward(self, values, conditions):
        conditions = _conditions(values, conditions, self.condition_dim)
        latent = self.encoder(torch.cat((values, conditions), dim=1))
        return self.decoder(torch.cat((latent, conditions), dim=1))


def _conditions(inputs, conditions, dimensions):
    if dimensions == 0:
        return inputs.new_empty((len(inputs), 0))
    conditions = torch.as_tensor(conditions, device=inputs.device, dtype=inputs.dtype)
    if conditions.ndim == 1 and dimensions == 1:
        conditions = conditions[:, None]
    if conditions.shape != (len(inputs), dimensions):
        raise ValueError('Expected one condition vector per LS')
    return conditions


def gaussian_probabilities(weights, means, scales, bins):
    """Gaussian-mixture mass integrated over native bins on the normalized axis.

    Each component is conditioned on falling inside the histogram range, so
    mixture weights describe the fractions inside that observed range.
    A tiny probability floor protects finite log likelihoods in distant tails.
    Survival-function differences avoid CDF cancellation above the means.
    """
    dtype = torch.float32 if means.device.type == 'mps' else torch.float64
    edges = torch.linspace(0, 1, bins + 1, device=means.device, dtype=dtype)
    z = (edges - means.to(dtype)[..., None]) / (scales.to(dtype)[..., None] * math.sqrt(2))
    lower, upper = z[..., :-1], z[..., 1:]
    if means.device.type == 'mps':
        # Reflect negative tails before subtracting survival functions. This
        # avoids cancellation in float32; MPS does not support float64.
        left = upper < 0
        first = torch.where(left, -upper, lower)
        last = torch.where(left, -lower, upper)
        masses = .5 * (torch.erfc(first) - torch.erfc(last))
    else:
        ordinary = .5 * (torch.erf(upper) - torch.erf(lower))
        survival = .5 * (torch.erfc(lower) - torch.erfc(upper))
        masses = torch.where(lower > 0, survival, ordinary)
    masses = masses.clamp_min(0)
    masses = masses / masses.sum(dim=-1, keepdim=True).clamp_min(1e-30)
    probabilities = (weights.to(dtype)[..., None] * masses).sum(dim=1).clamp_min(1e-12)
    return probabilities / probabilities.sum(dim=1, keepdim=True)


class HistogramMDN(nn.Module):
    """Gaussian-mixture shape decoder with a shared LS latent and pile-up.

    Inputs match preprocessing.prepare: native fractions + log-N + masks.
    autoencode=False predicts shapes from pile-up alone; conditional=False is
    an unconditioned shape AE. Neither variant reconstructs pile-up or counts.
    N=0 is handled by the likelihood/score masks, not interpreted as a shape.
    Phi is treated on its stored linear axis; this does not impose a wrapped
    periodic density across the -pi/pi boundary.
    """

    def __init__(self, bin_sizes, components=2, hidden=64, latent=4,
                 conditional=True, autoencode=True):
        super().__init__()
        self.bin_sizes = tuple(int(size) for size in bin_sizes)
        if (not self.bin_sizes or min(self.bin_sizes) < 1 or components < 1
                or hidden < 1 or latent < 1):
            raise ValueError('Expected positive bin, component and network sizes')
        self.components = int(components)
        self.input_features = sum(self.bin_sizes) + 2 * len(self.bin_sizes)
        self.condition_dim = int(bool(conditional))
        self.autoencode = bool(autoencode)
        if self.autoencode:
            self.encoder = nn.Sequential(
                nn.Linear(self.input_features + self.condition_dim, hidden), nn.SiLU(),
                nn.Linear(hidden, latent))
            self.decoder = nn.Sequential(nn.Linear(latent + self.condition_dim, hidden), nn.SiLU())
        else:
            self.encoder = None
            self.decoder = nn.Sequential(
                nn.Linear(max(1, self.condition_dim), hidden), nn.SiLU(),
                nn.Linear(hidden, hidden), nn.SiLU())
        self.heads = nn.ModuleList(nn.Linear(hidden, 3 * self.components) for _ in self.bin_sizes)

    def initialize_heads(self, pooled_counts):
        """Initialize using training-only pooled quantiles, with 0.5/bin smoothing."""
        if len(pooled_counts) != len(self.bin_sizes):
            raise ValueError('Expected one pooled training histogram per ME')
        with torch.no_grad():
            for bins, head, pooled in zip(self.bin_sizes, self.heads, pooled_counts):
                counts = torch.as_tensor(pooled, device=head.weight.device, dtype=torch.float64)
                if (counts.shape != (bins,) or not torch.isfinite(counts).all()
                        or (counts < 0).any()):
                    raise ValueError('Expected finite nonnegative pooled training counts')
                probabilities = (counts + .5) / (counts + .5).sum()
                centers = (torch.arange(bins, device=counts.device, dtype=torch.float64) + .5) / bins
                quantiles = (torch.arange(self.components, device=counts.device, dtype=torch.float64) + .5) / self.components
                locations = centers[torch.searchsorted(probabilities.cumsum(0), quantiles).clamp_max(bins - 1)]
                mean = (probabilities * centers).sum()
                std = torch.sqrt((probabilities * (centers - mean).square()).sum())
                floor = .5 / bins
                width = (std / math.sqrt(self.components)).clamp(min=floor + 1e-3, max=.45)
                scaled_width = ((width - floor) / (.5 - floor)).clamp(1e-4, 1 - 1e-4)
                head.weight.mul_(.02)
                bias = head.bias.reshape(3, self.components)
                bias[0].zero_()
                bias[1].copy_(torch.logit(locations.clamp(1e-4, 1 - 1e-4)))
                bias[2].fill_(float(torch.logit(scaled_width)))

    def forward(self, inputs, conditions):
        if inputs.ndim != 2 or inputs.shape[1] != self.input_features:
            raise ValueError('Expected normalized bins + log-N + nonempty masks')
        conditions = _conditions(inputs, conditions, self.condition_dim)
        if self.autoencode:
            latent = self.encoder(torch.cat((inputs, conditions), dim=1))
            hidden = self.decoder(torch.cat((latent, conditions), dim=1))
        else:
            hidden = self.decoder(conditions if self.condition_dim else inputs.new_zeros((len(inputs), 1)))
        predictions = []
        for bins, head in zip(self.bin_sizes, self.heads):
            logits, location, width = head(hidden).reshape(-1, 3, self.components).unbind(dim=1)
            weights, means = logits.softmax(dim=1), location.sigmoid()
            floor = .5 / bins
            scales = floor + (.5 - floor) * width.sigmoid()
            predictions.append(dict(probabilities=gaussian_probabilities(weights, means, scales, bins),
                                    weights=weights, means=means, scales=scales))
        return predictions


def _histogram_terms(predictions, histograms):
    if not predictions or len(predictions) != len(histograms):
        raise ValueError('Expected predictions and observed histograms for every ME')
    for prediction, histogram in zip(predictions, histograms):
        probabilities = prediction['probabilities']
        dtype = torch.float32 if probabilities.device.type == 'mps' else torch.float64
        probabilities = probabilities.to(dtype)
        counts = torch.as_tensor(histogram, device=probabilities.device, dtype=dtype)
        if counts.shape != probabilities.shape:
            raise ValueError('Predicted and observed histogram shapes differ')
        yield counts, probabilities, counts.sum(dim=1)


def histogram_loss(predictions, histograms, count_weighted=True):
    """Binned multinomial NLL, omitting data-only combinatorial constants.

    Average over observed MEs per LS, then observed LS. count_weighted=False
    instead averages per-count cross entropy, a deliberately different objective.
    Empty LS return None and contribute no gradient.
    """
    losses, observed = [], []
    for counts, probabilities, totals in _histogram_terms(predictions, histograms):
        value = -(counts * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        losses.append(value if count_weighted else value / totals.clamp_min(1))
        observed.append(totals > 0)
    valid, values = torch.stack(observed, dim=1), torch.stack(losses, dim=1)
    known = valid.any(dim=1)
    if not known.any():
        return None
    per_ls = torch.where(valid, values, 0).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    if per_ls.device.type == 'mps':
        # Boolean selection backward uses a nondeterministic MPS scatter.
        # Empty rows already contribute zero, so this is the same masked mean.
        return per_ls.sum() / known.sum()
    return per_ls[known].mean()


def histogram_scores(predictions, histograms):
    """Per-ME shape diagnostics; N=0 is NaN, not a zero anomaly score.

    Deviance and Pearson scores use counts and divide by native bins minus one.
    The Pearson denominator has a 0.5 expected-count floor for sparse bins.
    These are diagnostics, not chi-square p-values: bins share a normalization,
    predictions may reconstruct the observations, and sparse-bin asymptotics fail.
    """
    result = {key: [] for key in ('deviance', 'pearson', 'mse', 'nll_per_count')}
    for counts, probabilities, totals in _histogram_terms(predictions, histograms):
        expected = totals[:, None] * probabilities
        fractions = counts / totals.clamp_min(1)[:, None]
        divergence = 2 * (counts * (fractions.clamp_min(1e-12).log()
                                  - probabilities.clamp_min(1e-12).log())).sum(dim=1)
        terms = dict(
            deviance=divergence.clamp_min(0) / max(1, counts.shape[1] - 1),
            pearson=((counts - expected).square() / expected.clamp_min(.5)).sum(dim=1) / max(1, counts.shape[1] - 1),
            mse=(fractions - probabilities).square().mean(dim=1),
            nll_per_count=-(counts * probabilities.clamp_min(1e-12).log()).sum(dim=1) / totals.clamp_min(1))
        for key, value in terms.items():
            result[key].append(torch.where(totals > 0, value, torch.full_like(value, torch.nan)))
    return {key: torch.stack(value, dim=1) for key, value in result.items()}
