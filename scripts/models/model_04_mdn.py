"""Predict native-bin shapes with a Gaussian-mixture decoder and pile-up."""

from scripts.models.config import MDN
from scripts.models.networks import HistogramMDN

NAME = MDN
KIND = 'mdn'
SETTINGS = dict(components=8, hidden=64, latent=8, autoencode=True,
                epochs=100, lr=.0005, weight_decay=1e-5, patience=10,
                batch_size=1024, condition_transform='linear',
                count_weighted=False, score='deviance')


def build_model(dimensions, settings=None):
    settings = SETTINGS if settings is None else settings
    return HistogramMDN(dimensions, components=settings['components'],
                        hidden=settings['hidden'], latent=settings['latent'],
                        autoencode=settings['autoencode'], conditional=True)
