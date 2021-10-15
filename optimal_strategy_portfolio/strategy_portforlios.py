import numpy as np

from optimal_strategy_portfolio.strategies import Strategy


class StrategyPortfolio(object):
    def __init__(self, strats: [Strategy], trimmed_length=None):
        self.symbols = set([strat.symbol for strat in strats])
        self.n_strats = len(strats)
        self.strats = strats
        self.n_args = sum([strat.n_args for strat in strats])
        self.trimmed_length = trimmed_length or min([strat.trimmed_length for strat in strats])
        self._auto_trim_stats()
        self.returns = np.zeros(self.trimmed_length)
        self.reshaped_args = []

    def _auto_trim_stats(self):
        for strat in self.strats:
            strat.trimmed_length = self.trimmed_length

    def _reshape_args(self, args):
        k = 0
        if not self.reshaped_args:
            for strat in self.strats:
                self.reshaped_args += [tuple(args[k:k + strat.n_args])]
                k += strat.n_args
        else:
            for i, strat in enumerate(self.strats):
                self.reshaped_args[i] = tuple(args[k:k + strat.n_args])
                k += strat.n_args

        return self.reshaped_args

    def execute(self, args):
        reshaped_args = self._reshape_args(args)
        return np.array([strat.execute(*x) for strat, x in zip(self.strats, reshaped_args)])
