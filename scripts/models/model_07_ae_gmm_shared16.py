"""16 shared Gaussians per ME; keep the K8 experiment's other settings."""

from scripts.models.config import SHARED_GMM16_AE
from scripts.models.model_05_ae_gmm_shared import SETTINGS as K8_SETTINGS, build_model

NAME = SHARED_GMM16_AE
KIND = 'shared_gmm'
SETTINGS = dict(K8_SETTINGS, components=16)
