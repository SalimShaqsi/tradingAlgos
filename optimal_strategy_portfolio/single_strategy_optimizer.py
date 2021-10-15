import pandas as pd
from pypfopt import expected_returns, risk_models, base_optimizer

from .solver_wrappers import scipy_optimize_wrapper
from .strategies import Strategy
import numpy as np


class SingleStrategyOptimizer(object):
    def __init__(self, strategy: Strategy, bounds: np.ndarray, solver=scipy_optimize_wrapper,
                 **solver_kwargs):
        assert strategy.n_args > 0, "Strategy has no arguments to optimize."
        assert bounds.shape == (strategy.n_args, 2), "Bounds have the wrong shape"
        self.strategy = strategy
        self.bounds = bounds
        self.solver = solver
        self.solver_kwargs = solver_kwargs
        self.optimal_args, self.optimal_objective = None, None

    def objective_function(self, x):
        returns = self.strategy.execute(*x)
        returns = pd.DataFrame(returns)
        mu = expected_returns.ema_historical_return(returns, returns_data=True, compounding=False)
        S = risk_models.CovarianceShrinkage(returns, returns_data=True).ledoit_wolf()

        perf = base_optimizer.portfolio_performance(
            [1], mu, S, verbose=False, risk_free_rate=0.02
        )

        return -perf[2]  # negative sharpe ratio

    def run(self):
        self.optimal_objective, self.optimal_args, r = \
            self.solver(self.objective_function, self.bounds, **self.solver_kwargs)
        return self.optimal_args, self.optimal_objective




