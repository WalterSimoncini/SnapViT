from typing import Callable

from .enums import GALossType
from .losses import mse_loss, cosine_similarity_loss


def load_ga_loss(type_: GALossType) -> Callable:
    """Returns an initialized model of the given kind"""
    return {
        GALossType.MSE: mse_loss,
        GALossType.COSINE_SIMILARITY: cosine_similarity_loss,
    }[type_]
