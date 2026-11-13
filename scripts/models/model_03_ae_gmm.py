"""Fit each LS histogram with Gaussians, then reconstruct its shape parameters."""

from scripts.models.config import GMM_AE
from scripts.models.networks import ConditionalAE

NAME = GMM_AE
KIND = 'gmm'
SETTINGS = dict(components=8, hidden=(64, 8), epochs=100, lr=.001,
                weight_decay=1e-4, patience=10, batch_size=1024,
                condition_transform='linear', gmm_iterations=20,
                gmm_scale_floor_bins=12**-.5, gmm_pseudocount=.5)


def build_model(dimensions, settings=None):
    settings = SETTINGS if settings is None else settings
    return ConditionalAE(dimensions, hidden=tuple(settings['hidden']))
