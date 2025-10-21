import numpy as np

from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod

from .types import FitnessFunction


class GAOptimizer(ABC):
    def __init__(
        self,
        fitness_function: FitnessFunction,
        max_function_evaluations: int,
        ndim_problem: int,
        seed: int,
        **kwargs
    ):
        """
            Initialize a GA optimizer.

            Args:
                fitness_function: The fitness function to minimize.
                max_function_evaluations: The maximum number of function evaluations.
                ndim_problem: The number of dimensions of the problem.
                seed: The seed for the random number generator.
        """
        self.seed = seed
        self.ndim_problem = ndim_problem
        self.fitness_function = fitness_function
        self.max_function_evaluations = max_function_evaluations

    @abstractmethod
    def optimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
            Optimize the fitness function and return the best
            solution, its fitness and additional information
            as a dictionary.
        """
        raise NotImplementedError
