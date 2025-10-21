import numpy as np

from typing import Tuple, Dict, Any
from .enums import GAOptimizerType
from .optimizers import (
    XNESOptimizer,
    FitnessFunction
)


def optimize_function_ga(
    type_: GAOptimizerType,
    fitness_function: FitnessFunction,
    ndim_problem: int,
    max_function_evaluations: int = 250,
    seed: int = 42,
    **kwargs
) -> Tuple[np.ndarray, float, Dict[str, Any]]:
    """Optimizes the given fitness function using a genetic algorithm"""
    factory = {
        GAOptimizerType.XNES: XNESOptimizer
    }[type_]

    optimizer = factory(
        fitness_function=fitness_function,
        max_function_evaluations=max_function_evaluations,
        ndim_problem=ndim_problem,
        seed=seed,
        **kwargs
    )

    return optimizer.optimize()
