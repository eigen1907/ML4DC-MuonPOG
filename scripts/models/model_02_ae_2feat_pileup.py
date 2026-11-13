"""Reconstruct mean/std features, using pile-up only as a condition."""

from scripts.models.config import PILEUP_AE
from scripts.models.networks import ConditionalAE

NAME = PILEUP_AE
KIND = 'conditional'
SETTINGS = dict(hidden=(32, 4), epochs=120, lr=.0005, weight_decay=1e-4,
                patience=12, batch_size=1024, condition_transform='linear', use_pileup=True)


def build_model(dimensions, settings=None):
    settings = SETTINGS if settings is None else settings
    return ConditionalAE(dimensions, hidden=tuple(settings['hidden']),
                         condition_dim=int(settings.get('use_pileup', True)))
