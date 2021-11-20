import pandas as pd
from pypfopt import expected_returns, risk_models, base_optimizer

from .metrics import sharpe, ew_mean, sortino
from .solver_wrappers import scipy_optimize_wrapper, pygad_wrapper
import numpy as np


class SingleStrategyOptimizer(object):
    def __init__(self, strategy, bounds=None, solver=pygad_wrapper, coerce=False,
                 **solver_kwargs):
        assert strategy.n_args > 0, "Strategy has no arguments to optimize."
        self.strategy = strategy
        self.bounds = bounds if bounds is not None else strategy.bounds
        assert np.array(self.bounds).shape == (strategy.n_args, 2), "Bounds have the wrong shape"
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        if solver == pygad_wrapper and not coerce:
            self.solver_kwargs['gene_type'] = list(strategy.arg_types)

        self.optimal_args, self.optimal_objective = None, None
        self.coerce = coerce

    def objective_function(self, x):
        returns = self.strategy.execute(*x)
        return -sortino(returns, mean_func=ew_mean)  # negative sharpe ratio

    def coerced_objective_function(self, x):
        x = [t(a) for a, t in zip(x, self.strategy.arg_types)]  # coerce args into correct types
        returns = self.strategy.execute(*x)
        return -sortino(returns, mean_func=ew_mean)  # negative sharpe ratio

    def run(self):
        objective_function = self.coerced_objective_function if self.coerce else self.objective_function
        self.optimal_objective, self.optimal_args, r = \
            self.solver(objective_function, self.bounds, **self.solver_kwargs)
        return self.optimal_args, self.optimal_objective




