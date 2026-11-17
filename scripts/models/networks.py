"""Small fully connected autoencoder for ME features or Gaussian weights."""

import torch
from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden=(32, 4)):
        super().__init__()
        widths = (int(input_dim), *(int(n) for n in hidden))
        if len(widths) < 2 or min(widths) < 1:
            raise ValueError('Expected positive input and hidden sizes')
        encoder = []
        for index, (before, after) in enumerate(zip(widths[:-1], widths[1:])):
            encoder.append(nn.Linear(before, after))
            if index < len(widths) - 2:
                encoder.append(nn.SiLU())
        decoder = []
        reversed_widths = tuple(reversed(widths))
        for index, (before, after) in enumerate(zip(reversed_widths[:-1], reversed_widths[1:])):
            decoder.append(nn.Linear(before, after))
            if index < len(widths) - 2:
                decoder.append(nn.SiLU())
        self.encoder = nn.Sequential(*encoder)
        self.decoder = nn.Sequential(*decoder)

    def forward(self, values):
        return self.decoder(self.encoder(values))


class MaskedFreeAutoEncoder(nn.Module):
    def __init__(self, dimensions, components=16, hidden=(64, 8)):
        super().__init__()
        self.block_size = 3*int(components)
        self.n_mes = int(dimensions)//self.block_size
        if self.n_mes < 2 or self.n_mes*self.block_size != dimensions or len(hidden) != 2:
            raise ValueError('Expected complete free-Gaussian ME blocks and two hidden widths')
        wide, latent = hidden
        self.encoder = nn.Sequential(nn.Linear(dimensions+self.n_mes, wide), nn.SiLU(), nn.Linear(wide, latent))
        self.decoder = nn.Sequential(nn.Linear(latent, wide), nn.SiLU(), nn.Linear(wide, dimensions))

    def forward(self, values, observed, target_me):
        if values.ndim != 2 or values.shape[1] != self.n_mes*self.block_size:
            raise ValueError('Expected one free-Gaussian block per ME')
        if observed.shape != (len(values), self.n_mes) or observed.dtype != torch.bool:
            raise ValueError('Expected one boolean observation flag per ME')
        target = torch.as_tensor(target_me, dtype=torch.long, device=values.device)
        if target.ndim == 0:
            target = target.expand(len(values))
        if target.shape != (len(values),) or (target < 0).any() or (target >= self.n_mes).any():
            raise ValueError('Expected one valid target ME per observation')
        visible = observed.clone()
        visible[torch.arange(len(values), device=values.device), target] = False
        blocks = values.reshape(-1, self.n_mes, self.block_size)
        masked = torch.where(visible[:, :, None], blocks, 0).flatten(1)
        return self.decoder(self.encoder(torch.cat((masked, visible.to(values.dtype)), 1)))
