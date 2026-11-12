"""Single-component DM histogram AE with variance-weighted residuals."""

from scripts.models.config import DM_AE
from scripts.models.core import HistogramAE

NAME = DM_AE
KIND = 'histogram'
DISTRIBUTION = 'dm'
SCORE = 'weighted'
SOURCE = WINDOW = None


def build_model(bin_sizes):
    return HistogramAE(bin_sizes, DISTRIBUTION)
