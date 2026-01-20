from enum import Enum


class GALossType(str, Enum):
    MSE = "mse"
    COSINE_SIMILARITY = "cosine-similarity"
