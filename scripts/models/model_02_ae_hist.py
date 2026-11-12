"""Native-bin histogram autoencoder with mean-squared reconstruction loss."""

from scripts.models.config import HIST_AE
from scripts.models.core import HistogramAE

NAME = HIST_AE
KIND = 'histogram'
DISTRIBUTION = SCORE = 'mse'
SOURCE = WINDOW = None


def build_model(bin_sizes):
    return HistogramAE(bin_sizes, DISTRIBUTION)
