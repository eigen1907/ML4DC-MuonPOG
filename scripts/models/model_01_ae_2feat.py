"""Mean/std feature autoencoder trained on DCS-ready Good LS."""

from sklearn.neural_network import MLPRegressor

from scripts.models.config import FEATURE_AE, SEED
from scripts.models.core import FEATURE_EPOCHS, BATCH_SIZE, HIDDEN_LAYERS

NAME = FEATURE_AE
KIND = 'feature'
DISTRIBUTION = SCORE = SOURCE = WINDOW = None


def build_model():
    return MLPRegressor(hidden_layer_sizes=HIDDEN_LAYERS, batch_size=BATCH_SIZE,
                        random_state=SEED, tol=0, n_iter_no_change=FEATURE_EPOCHS + 1)
