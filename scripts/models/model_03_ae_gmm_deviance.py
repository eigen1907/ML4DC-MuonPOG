"""Masked free16 GMM AE trained on raw Input–Predict count deviance."""

from scripts.models.networks import MaskedFreeAutoEncoder

from scripts.models.config import GMM_DEVIANCE
from scripts.models.gaussian import SCALE_FLOOR_BINS


NAME = GMM_DEVIANCE
KIND = 'gmm'
SETTINGS = dict(components=16, hidden=(64, 8), epochs=100, lr=.001,
                weight_decay=0., patience=10, batch_size=1024,
                basis_iterations=80, reference_iterations=150,
                weight_iterations=60, scale_floor_bins=float(SCALE_FLOOR_BINS),
                loss='input_histogram_deviance', masked=True, free=True, prefit_iterations=60,
                prefit_lr=.04, prefit_batch_size=2048, prefit_device='mps',
                prefit_tolerance=1e-6, max_scale=.5)


def build_model(dimensions, settings=None):
    settings = SETTINGS if settings is None else settings
    return MaskedFreeAutoEncoder(dimensions, settings['components'], tuple(settings['hidden']))
