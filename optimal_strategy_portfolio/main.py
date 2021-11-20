from timeit import timeit

import pandas as pd
from numba import njit

from optimal_strategy_portfolio.metrics import annualized_sharpe, sharpe
from optimal_strategy_portfolio.signals.aggregators import HProd, HMean
from optimal_strategy_portfolio.signals.price_signals import PriceMASignal, PriceEMASignal, PMA, R, V
from optimal_strategy_portfolio.signals.transforms import SMA, EMA
from optimal_strategy_portfolio.signals.variables import BoolVar, IntVar
from shared_utils.data import get_securities_data, set_universe
import numpy as np



@njit
def roll(a, w):
    l = len(a)
    return [a[k:n] for k, n in zip(range(l - w + 1), range(w, l + 1))]

@njit
def roll2(a, w):
    return np.lib.stride_tricks.sliding_window_view(a, w)


if __name__ == '__main__':
    symbols = ["FB", "GOOG", "AAPL", "MSFT", "TSLA"]

    set_universe(symbols, test_start_date='2019-01-01')

    fast_mas = [PriceEMASignal(symb) for symb in symbols]
    slow_mas = [PriceEMASignal(symb) for symb in symbols]

    returns = [R(symb) for symb in symbols]

    returns_mas = [EMA(r) for r in returns]

    volumes = [V(symb) for symb in symbols]

    volume_mas = [EMA(v) for v in volumes]

    h_mean_returns = HMean(returns_mas)

    abs_buy_signals = [fast > slow for fast, slow in zip(fast_mas, slow_mas)]
    abs_sell_signals = [fast < slow for fast, slow in zip(fast_mas, slow_mas)]

    rel_buy_signals = [r > h_mean_returns for r in returns_mas]
    rel_sell_signals = [r < h_mean_returns for r in returns_mas]

    buy_signals = [(BoolVar() | a_buy) * (BoolVar() | r_buy) | (BoolVar() * r_buy) for a_buy, r_buy in
                   zip(abs_buy_signals, rel_buy_signals)]
    sell_signals = [(BoolVar() | a_sell) * (BoolVar() | r_sell) | (BoolVar() * r_sell) for a_sell, r_sell in
                    zip(abs_sell_signals, rel_sell_signals)]

    signals = [buy - sell for buy, sell in
               zip(buy_signals, sell_signals)]

    strats = [s.to_strategy() for s in signals]

    for strat in strats:
        strat.optimize(num_generations=500).test()

    sharpes = [annualized_sharpe(strat.test_results) for strat in strats]
    print({symb: sharpe for symb, sharpe in zip(symbols, sharpes)})

    print({symb: strat.optimization_results[0] for strat, symb, sharpe in
           zip(strats, symbols, sharpes)})








