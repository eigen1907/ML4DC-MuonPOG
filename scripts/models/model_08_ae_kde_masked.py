"""Predict each original ME from the other KDE shapes and pile-up.

Training masks one randomly chosen observed ME per LS and fits only that ME's
bins. Inference holds out each ME in turn, so no predicted block sees its own
input. Availability flags distinguish an observed zero from a missing shape.
"""

import torch
from torch import nn

from scripts.models.config import KDE_MASKED_AE
from scripts.models.model_06_ae_kde import SETTINGS as KDE_SETTINGS
from scripts.models.networks import _conditions


NAME = KDE_MASKED_AE
KIND = 'kde'
SETTINGS = dict(KDE_SETTINGS, mask_me=True)


class MaskedConditionalAE(nn.Module):
    """Small conditional AE with complete ME blocks removed before encoding."""

    def __init__(self, bin_sizes, hidden=(64, 8), condition_dim=1):
        super().__init__()
        self.bin_sizes = tuple(int(size) for size in bin_sizes)
        widths = tuple(int(width) for width in hidden)
        if (not self.bin_sizes or min(self.bin_sizes) < 1 or not widths
                or min(widths) < 1 or condition_dim < 0):
            raise ValueError('Expected positive bin/hidden sizes and a nonnegative condition size')
        self.input_dim = sum(self.bin_sizes)
        self.me_count, self.condition_dim = len(self.bin_sizes), int(condition_dim)
        self.register_buffer('feature_me', torch.repeat_interleave(
            torch.arange(self.me_count), torch.tensor(self.bin_sizes)), persistent=False)
        encoder, previous = [], self.input_dim + self.me_count + self.condition_dim
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

    def _observed(self, values, observed):
        if values.ndim != 2 or values.shape[1] != self.input_dim:
            raise ValueError('Expected one concatenated KDE-bin vector per LS')
        if observed is None:
            return torch.ones((len(values), self.me_count), dtype=torch.bool, device=values.device)
        observed = torch.as_tensor(observed, dtype=torch.bool, device=values.device)
        if observed.shape != (len(values), self.me_count):
            raise ValueError('Expected one observed/nonempty flag per LS and ME')
        return observed

    def sample_target(self, observed):
        """Choose one available target ME per row using the seeded torch RNG.

        All-empty rows return target 0; callers must omit these rows from loss.
        Pass the returned vector to forward and score only its target bins.
        """
        observed = torch.as_tensor(observed, dtype=torch.bool, device=self.feature_me.device)
        if observed.ndim != 2 or observed.shape[1] != self.me_count:
            raise ValueError('Expected one observed/nonempty flag per LS and ME')
        weights = observed.float()
        weights = torch.where(observed.any(dim=1, keepdim=True), weights, torch.ones_like(weights))
        targets = torch.multinomial(weights, 1).squeeze(1)
        return torch.where(observed.any(dim=1), targets, torch.zeros_like(targets))

    def forward(self, values, conditions, observed=None, target_me=None):
        """Return raw-bin predictions with target inputs and flags zeroed.

        target_me is one ME index for the batch or one index per LS. Training
        uses sample_target(observed) explicitly so the loss can select the same
        targets. With no target specified, assemble all leave-one-ME-out blocks.
        Inputs may be standardized: masking happens here, after preprocessing.
        Caller tensors are never changed.
        """
        observed = self._observed(values, observed)
        conditions = _conditions(values, conditions, self.condition_dim)
        if target_me is None:
            return self.predict_heldout(values, conditions, observed)
        targets = torch.as_tensor(target_me, dtype=torch.long, device=values.device)
        if targets.ndim == 0:
            targets = targets.expand(len(values))
        if targets.shape != (len(values),) or ((targets < 0) | (targets >= self.me_count)).any():
            raise ValueError('Expected one valid held-out ME index per LS')
        available = observed & (torch.arange(self.me_count, device=values.device)[None] != targets[:, None])
        masked = torch.where(available[:, self.feature_me], values, torch.zeros_like(values))
        latent = self.encoder(torch.cat((masked, available.to(values.dtype), conditions), dim=1))
        return self.decoder(torch.cat((latent, conditions), dim=1))

    def predict_heldout(self, values, conditions, observed=None):
        """Assemble each ME only from its own deterministic held-out pass.

        Use ordinary inference batches; sequential passes keep temporary memory
        to one batch instead of expanding it by the number of MEs. This method
        leaves train/eval and gradient contexts to the caller.
        """
        observed = self._observed(values, observed)
        result, start = [], 0
        for me, size in enumerate(self.bin_sizes):
            prediction = self(values, conditions, observed, target_me=me)
            # Copy the block so a view does not retain all eight full outputs.
            result.append(prediction[:, start:start + size].clone())
            start += size
        return torch.cat(result, dim=1)


def build_model(dimensions, settings=None):
    """dimensions is native bin sizes, or a total with settings['bin_sizes']."""
    settings = SETTINGS if settings is None else settings
    if isinstance(dimensions, int):
        bin_sizes = settings.get('bin_sizes')
        if bin_sizes is None or sum(bin_sizes) != dimensions:
            raise ValueError('Masked KDE requires native bin_sizes to identify complete MEs')
    else:
        bin_sizes = dimensions
    return MaskedConditionalAE(bin_sizes, hidden=tuple(settings['hidden']))
