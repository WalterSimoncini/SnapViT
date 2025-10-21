import numpy as np

from pypop7.optimizers.nes.xnes import XNES


class CustomXNES(XNES):
    """
        Customized version of the XNES algorithm that allows
        for a user-specified initialization of the covariance
        matrix.
    """
    def initialize(self, is_restart=False):
        x = np.empty((self.n_individuals, self.ndim_problem))  # offspring population
        y = np.empty((self.n_individuals,))  # fitness (no evaluation)
        mean = self._initialize_mean(is_restart)  # mean of Gaussian search distribution

        if self.options.get("covariance_init", None) is not None:
            covariance_init = self.options.get("covariance_init")
        else:
            covariance_init = np.eye(self.ndim_problem)

        a = np.linalg.cholesky(covariance_init)
        inv_a = np.linalg.inv(a)
        log_det = np.log(np.linalg.det(a)) * 2

        self._w = np.maximum(0.0, np.log(self.n_individuals/2.0 + 1.0) - np.log(
            self.n_individuals - np.arange(self.n_individuals)))

        # Initialize an empty list to store fitness scores
        self.fitness_scores = []

        return x, y, mean, a, inv_a, log_det

    def iterate(self, x=None, y=None, mean=None, a=None, args=None):
        self.covariance = a @ a.T

        for k in range(self.n_individuals):
            if self._check_terminations():
                return x, y

            x[k] = mean + np.dot(a, self.rng_optimization.standard_normal((self.ndim_problem,)))
            y[k] = self._evaluate_fitness(x[k], args)

            # Store the list of fitness values
            self.fitness_scores.append(y[k])

        return x, y

    def _collect(self, fitness=None, y=None, mean=None):
        results = super()._collect(fitness, y, mean)

        results["fitness_scores"] = self.fitness_scores
        results["covariance"] = self.covariance.tolist()

        return results
