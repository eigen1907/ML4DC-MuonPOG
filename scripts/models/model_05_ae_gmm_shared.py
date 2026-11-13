"""Fixed Gaussian shapes; reconstruct LS mixture weights and score original bins."""

from scripts.models.config import SHARED_GMM_AE
from scripts.models.networks import ConditionalAE

NAME = SHARED_GMM_AE
KIND = 'shared_gmm'
SETTINGS = dict(components=30, hidden=(64, 8), epochs=100, lr=.001,
                weight_decay=1e-4, patience=10, batch_size=1024,
                condition_transform='linear', basis_iterations=80,
                prior_strength=8., weight_iterations=30,
                score='variance_residual')


def build_model(dimensions, settings=None):
    settings = SETTINGS if settings is None else settings
    return ConditionalAE(dimensions, hidden=tuple(settings['hidden']))
