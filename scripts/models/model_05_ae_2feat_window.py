"""Model 01 scores averaged over five adjacent LS; no independent fitting."""

from scripts.models.config import FEATURE_AE, FEATURE_WINDOW_AE, WINDOW_SIZE
from scripts.models.model_01_ae_2feat import build_model

NAME = FEATURE_WINDOW_AE
KIND = 'window'
DISTRIBUTION = SCORE = None
SOURCE = FEATURE_AE
WINDOW = WINDOW_SIZE
