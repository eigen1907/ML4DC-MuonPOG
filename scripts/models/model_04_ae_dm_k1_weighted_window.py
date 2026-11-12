"""Model 03 scores averaged over five adjacent LS; no independent fitting."""

from scripts.models.config import DM_AE, WINDOW_AE, WINDOW_SIZE
from scripts.models.model_03_ae_dm_k1 import build_model

NAME = WINDOW_AE
KIND = 'window'
DISTRIBUTION = 'dm'
SCORE = 'weighted'
SOURCE = DM_AE
WINDOW = WINDOW_SIZE
