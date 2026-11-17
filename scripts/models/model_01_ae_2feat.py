"""Mean/std autoencoder using signed histogram axes, without external inputs."""

from scripts.models.config import TWO_FEATURE_AE
from scripts.models.networks import AutoEncoder

NAME = TWO_FEATURE_AE
KIND = 'moments'
SETTINGS = dict(hidden=(32, 4), epochs=120, lr=.0005, weight_decay=1e-4,
                patience=12, batch_size=1024, moment_order=2)


def build_model(dimensions, settings=None):
    settings = SETTINGS if settings is None else settings
    return AutoEncoder(dimensions, hidden=tuple(settings['hidden']))
