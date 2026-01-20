from enum import Enum


class GAOptimizerType(str, Enum):
    XNES = "xnes"
    PYGAD = "pygad"
