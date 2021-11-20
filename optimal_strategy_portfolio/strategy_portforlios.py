from timeit import timeit
from collections.abc import Iterable

import numpy as np

from optimal_strategy_portfolio.strategies import Strategy, PriceMovingAverageCrossover
from shared_utils.data import get_securities_data


class StrategyPortfolio(object):
    def __init__(self, strat_structure: [list, tuple, np.ndarray], trimmed_length: int = None):
        self.strat_structure = strat_structure
        self.arg_structure = []
        self._check_structure()
        self.reshaped_args = []
        self.n_args = sum([strats[0].n_args for strats in self.strat_structure])
        self._init_arg_structure()
        self.trimmed_length = trimmed_length or min([strat.trimmed_length for strat in self.strats])
        self.returns = np.zeros(self.trimmed_length)
        self.bounds = ()
        self.arg_types = ()
        self.arg_names = ()
        for strats in self.strat_structure:
            self.bounds += strats[0].bounds
            self.arg_types += strats[0].arg_types

        if all([hasattr(strat, 'test_instance') for strat in self.strats]):
            test_structure = []
            for strats in self.strat_structure:
                test_structure += [[strat.test_instance for strat in strats]]
            self.test_instance = self.__class__(test_structure)

    def _check_structure(self):
        if type(self.strat_structure) not in [list, np.ndarray]:
            raise TypeError("Strat structure must either be sequences containing "
                            "strategies or sequences of strategies")

        for s in self.strat_structure:
            if type(s) not in [list, np.ndarray] and not isinstance(s, Strategy):
                raise TypeError("Strat structure must either be sequences containing "
                                "strategies or sequences of strategies")
        self.strats = []
        self.strat_structure = \
            [[s, [s]][isinstance(s, Strategy)] for s in self.strat_structure]  # convert strategies to lists of strats
        for strats in self.strat_structure:
            assert len({s.n_args for s in strats}) == 1, \
                "All linked strategies must have the same argument length"
            self.strats += strats
        self.n_strats = len(self.strats)

    def _init_arg_structure(self):
        args = [0] * self.n_args  # dummy args to initialize reshaped args array

        k = 0
        for strats in self.strat_structure:
            self.arg_structure += [(strats[0].n_args, len(strats))]
            n_args = strats[0].n_args
            for i in range(len(strats)):
                self.reshaped_args += [args[k:k + n_args]]
            k += n_args

    def _reshape_args(self, args: [list, tuple, np.ndarray]):
        j = 0
        k = 0
        for n_args, n in self.arg_structure:
            for x in range(n):
                self.reshaped_args[j] = args[k:k + n_args]
                j += 1
            k += n_args

        return self.reshaped_args

    def _reshape_args_old(self, args: [list, tuple, np.ndarray]):
        k = 0
        if not self.reshaped_args:
            for strats in self.strat_structure:
                n_args = strats[0].n_args
                for i in range(len(strats)):
                    self.reshaped_args += [tuple(args[k:k + n_args])]
                k += n_args
        else:
            for strats in self.strat_structure:
                n_args = strats[0].n_args
                for i in range(len(strats)):
                    self.reshaped_args[i] = args[k:k + n_args]
                k += n_args

        return self.reshaped_args

    def execute(self, args: [list, tuple, np.ndarray]):
        reshaped_args = self._reshape_args(args)
        return np.array([strat.execute(*x)[-self.trimmed_length:] for strat, x in zip(self.strats, reshaped_args)])

    def test(self, args):
        portfolio = self.test_instance if hasattr(self, 'test_instance') else self
        if not args and self.optimization_results:
            self.test_results = portfolio.execute(self.optimization_results[0])
            return self
        else:
            return portfolio.execute(args)

    def execute_with_weights(self, args: [list, tuple, np.ndarray], weights: np.ndarray):
        reshaped_args = self._reshape_args(args)
        return weights @ np.array(
            [strat.execute(*x)[-self.trimmed_length:] for strat, x in zip(self.strats, reshaped_args)])

    def test_with_weight(self, args: [list, tuple, np.ndarray], weights: np.ndarray):
        return self.test_instance.execute_with_weights(args, weights)

    def __add__(self, other):
        if isinstance(other, Strategy):
            return StrategyPortfolio(self.strat_structure + [[other]])
        elif isinstance(other, StrategyPortfolio):
            return StrategyPortfolio(self.strat_structure + other.strat_structure)
        else:
            raise TypeError(f"+ operator with type {type(other)} is not supported")

    def __radd__(self, other):
        if isinstance(other, Strategy):
            return StrategyPortfolio([[other]] + self.strat_structure)
        elif isinstance(other, StrategyPortfolio):
            return StrategyPortfolio(other.strat_structure + self.strat_structure)
        else:
            raise TypeError(f"+ operator with type {type(other)} is not supported")

    def __matmul__(self, other):
        if isinstance(other, Strategy):
            if len(self.strat_structure) != 1:
                raise ValueError(f"Structure shapes don't match: {len(self.strat_structure)}, 1")
            return StrategyPortfolio([strats + [other] for strats in self.strat_structure])
        elif isinstance(other, StrategyPortfolio):
            if len(self.strat_structure) != len(other.strat_structure):
                raise ValueError(f"Structure shapes don't match: {len(self.strat_structure)}, "
                                 f"{len(other.strat_structure)}")
            return StrategyPortfolio([strats + other_strats for strats, other_strats in
                                      zip(self.strat_structure, other.strat_structure)])
        else:
            raise TypeError(f"@ operator with type {type(other)} is not supported")


if __name__ == '__main__':
    price_data = get_securities_data(['AMZN'])['AMZN']
    strat = PriceMovingAverageCrossover(price_data)

    strats_ = [strat] * 10

    port = StrategyPortfolio(strats_)

    args_ = [50, 200] * 10

    t1 = timeit('port._reshape_args_old(args_)', number=100_000, globals=globals())
    t2 = timeit('port._reshape_args(args_)', number=100_000, globals=globals())
    t3 = timeit('port.execute(args_)', number=100_000, globals=globals())

    print(t1, t2, t3)
