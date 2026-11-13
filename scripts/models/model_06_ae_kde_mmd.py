"""Reuse the trained KDE AE; calibrate MMD scores without retraining weights."""

from scripts.models.config import KDE_MMD_AE
from scripts.models.model_06_ae_kde import SETTINGS as KDE_SETTINGS, build_model

NAME = KDE_MMD_AE
KIND = 'kde'
SETTINGS = dict(KDE_SETTINGS, score='mmd', mmd_bandwidth_bins=(1., 4., 16.),
                mmd_count_corrected=True, mmd_top_n=None)
