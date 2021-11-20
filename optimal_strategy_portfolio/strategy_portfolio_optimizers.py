import pandas as pd
from numba import njit
import numpy as np

from optimal_strategy_portfolio.metrics import sortino, ew_mean
from optimal_strategy_portfolio.solver_wrappers import pypfopt_ef_wrapper, scipy_optimize_wrapper, pygad_wrapper
from optimal_strategy_portfolio.strategies import Strategy
from optimal_strategy_portfolio.strategy_portforlios import StrategyPortfolio
from shared_utils.data import returns_from_prices


class StrategyPortfolioOptimizer:
    def __init__(self, strategy_portfolio: StrategyPortfolio, weight_bounds=(0, 1),
                 solver=pygad_wrapper, coerce=False, **solver_kwargs):
        self.strategy_portfolio = strategy_portfolio
        self.bounds = strategy_portfolio.bounds
        self._set_weight_bounds(weight_bounds)
        self.arg_types = strategy_portfolio.arg_types + (float,) * strategy_portfolio.n_strats
        self.solver = solver
        self.solver_kwargs = solver_kwargs or dict()
        self.optimal_args = self.optimal_objective = None, None

        if solver == pygad_wrapper and not coerce:
            self.solver_kwargs['gene_type'] = list(self.arg_types)

    def _set_weight_bounds(self, weight_bounds):
        if type(weight_bounds[0]) in {tuple, list, np.ndarray}:
            self.bounds += weight_bounds

    def objective_function(self, x):
        n_strats = self.strategy_portfolio.n_strats
        returns = self.strategy_portfolio.execute_with_weights(
            x[:-n_strats],
            x[n_strats:]
        )

        return sortino(returns, mean_func=ew_mean)

    def run(self):
        self.optimal_objective, self.optimal_args, r = \
            self.solver(self.objective_function, self.bounds, **self.solver_kwargs)
        return self.optimal_args, self.optimal_objective


class BilevelStrategyPortfolioOptimizer(object):
    def __init__(self, strategy_portfolio: StrategyPortfolio, bounds, portfolio_optimizer=pypfopt_ef_wrapper,
                 solver=pygad_wrapper, portfolio_optimizer_kwargs=None, solver_kwargs=None):
        assert len(bounds) == strategy_portfolio.n_args, "Length of bounds must equal the length of arguments"
        self.strategy_portfolio = strategy_portfolio
        self.bounds = bounds
        self.portfolio_optimizer = portfolio_optimizer
        self.solver = solver
        self.portfolio_optimizer_kwargs = portfolio_optimizer_kwargs or dict()
        self.solver_kwargs = solver_kwargs or dict()
        self.optimal_args, self.optimal_objective = None, None

    def objective_function(self, x):
        returns = self.strategy_portfolio.execute(x)
        returns = pd.DataFrame(np.transpose(returns), columns=[strat.name for strat in self.strategy_portfolio.strats])

        return self.portfolio_optimizer(returns, **self.portfolio_optimizer_kwargs)

    def run(self):
        self.optimal_objective, self.optimal_args, r = \
            self.solver(self.objective_function, self.bounds, **self.solver_kwargs)
        return self.optimal_args, self.optimal_objective

    def get_optimal_weights(self, x):
        returns = self.strategy_portfolio.execute(x)
        returns = pd.DataFrame(np.transpose(returns), columns=[strat.name for strat in self.strategy_portfolio.strats])

        return self.portfolio_optimizer(returns, return_weights=True)







