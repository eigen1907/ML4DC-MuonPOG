"""Spatially adaptive KDE inputs; original native-bin reconstruction targets."""

from scripts.models.config import KDE_ADAPTIVE_AE
from scripts.models.model_06_ae_kde import SETTINGS as KDE_SETTINGS, build_model

NAME = KDE_ADAPTIVE_AE
KIND = 'kde'
SETTINGS = dict(KDE_SETTINGS, adaptive=True, adaptive_multipliers=(.5, 1., 2.),
                adaptive_clip=(.5, 2.), adaptive_cv_ls=256, adaptive_cv_prior=8.,
                adaptive_seed=42)
