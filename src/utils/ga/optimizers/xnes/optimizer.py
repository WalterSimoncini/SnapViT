import logging
import numpy as np

from typing import Tuple, Dict, Any

from src.utils.ga.optimizers.base import GAOptimizer
from src.utils.ga.optimizers.types import FitnessFunction

from .algorithm import CustomXNES


class XNESOptimizer(GAOptimizer):
    def __init__(
        self,
        fitness_function: FitnessFunction,
        max_function_evaluations: int,
        ndim_problem: int,
        seed: int,
        lower_boundary: float = 0.0,
        upper_boundary: float = 2.0,
        starting_point: np.ndarray = None,
        covariance_init: np.ndarray = None,
        xnes_num_individuals: int = None,
        **kwargs
    ):
        super().__init__(
            fitness_function=fitness_function,
            max_function_evaluations=max_function_evaluations,
            ndim_problem=ndim_problem,
            seed=seed,
            **kwargs
        )

        self.lower_boundary = lower_boundary
        self.upper_boundary = upper_boundary
        self.covariance_init = covariance_init
        self.xnes_num_individuals = xnes_num_individuals

        if starting_point is None:
            self.starting_point = np.ones(self.ndim_problem)
        else:
            self.starting_point = starting_point

    def optimize(self) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        problem = {
            "fitness_function": self.fitness_function,
            "ndim_problem": self.ndim_problem,
            "lower_boundary": self.lower_boundary * np.ones(self.ndim_problem),
            "upper_boundary": self.upper_boundary * np.ones(self.ndim_problem)
        }

        options = {
            "max_function_evaluations": self.max_function_evaluations,
            "seed_rng": self.seed,
            "mean": self.starting_point,
            "covariance_init": self.covariance_init
        }

        if self.xnes_num_individuals is not None:
            options["n_individuals"] = self.xnes_num_individuals

            logging.info(f"using {self.xnes_num_individuals} individuals for XNES")
        else:
            logging.info("using the default number of individuals for XNES")

        logging.info(f"XNES options: {options}")

        solution = CustomXNES(problem, options).optimize()

        return (
            solution["best_so_far_x"].astype(np.float32),
            solution["best_so_far_y"],
            {
                "fitness_scores": solution["fitness_scores"],
                "covariance": solution["covariance"]
            }
        )
