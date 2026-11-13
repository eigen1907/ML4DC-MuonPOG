"""KDE-preprocessed LS inputs; reconstruct the original native-bin fractions."""

from scripts.models.config import KDE_AE
from scripts.models.networks import ConditionalAE

NAME = KDE_AE
KIND = 'kde'
SETTINGS = dict(hidden=(64, 8), epochs=100, lr=.001, weight_decay=1e-4,
                patience=10, batch_size=1024, condition_transform='linear',
                bandwidth_bins=1., prior_strength=8., score='variance_residual')


def build_model(dimensions, settings=None):
    settings = SETTINGS if settings is None else settings
    return ConditionalAE(dimensions, hidden=tuple(settings['hidden']))
