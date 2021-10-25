import pandas as pd
import numpy as np
from numba import jit

import matplotlib.pyplot as plt
from inspect import signature

from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, base_optimizer
from pypfopt import expected_returns
from pypfopt import objective_functions

from random import randint

from scipy import optimize

import pygad

import quantstats as qs

from scipy.sparse.linalg.eigen.arpack.arpack import ArpackNoConvergence
from pypfopt.exceptions import OptimizationError

from shared_utils.data import moving_average4, get_data


def annual_sharpe(prices, rf=0.0):
    returns = qs.utils.to_returns(prices, rf=rf)
    return qs.stats.sharpe(returns, rf=rf)


def moving_average_crossover_strat(s, fast_window_size, slow_window_size):
    s = np.array(s)
    s = np.insert(s, 0, 0)
    fast_window_size, slow_window_size = int(fast_window_size) or 1, int(slow_window_size) or 1
    slow_mv = moving_average4(s, slow_window_size)
    fast_mv = moving_average4(s, fast_window_size)

    min_len = min(len(slow_mv), len(fast_mv))

    positions = ((fast_mv[-min_len:] - slow_mv[-min_len:]) > 0) * 1

    return positions


def buy_and_hold_strat(s):
    return np.ones(len(s))


def prices_from_positions(s, positions):
    n = positions.shape[0]
    return [(positions[i, 1:] * np.diff(s) / s[1:] + 1) * s[0] for i in range(n)]


@jit(nopython=True)
def returns_from_positions(s, positions):
    n = positions.shape[0]
    return [(positions[i, :-1] * np.diff(s) / s[1:]) for i in range(n)]


strat_input_lengths = {
    strat: len(signature(strat).parameters) - 1 for strat in [moving_average_crossover_strat, buy_and_hold_strat]
}


def get_n_inputs(strats, n_s):
    return sum([strat_input_lengths[strat] for strat in strats]) * n_s


def trim_positions(positions):
    min_length = min([len(p) for p in positions])

    return np.array([p[-min_length:] for p in positions])


def split_inputs(strats, inputs, n_s):
    # take  flattened inputs and create x list of tuples containing inputs for each strat
    k = 0
    fixed_inputs = []
    for i in range(n_s):
        for strat in strats:
            input_length = strat_input_lengths[strat]
            fixed_inputs += [[i for i in inputs[k:k + input_length]]]
            k += input_length

    return fixed_inputs


def build_returns(inputs, n_s, price_data, strats, keep_index=False):
    returns = []
    inputs = split_inputs(strats, inputs, n_s)
    for p in price_data:
        s = price_data[p] if type(price_data) in (pd.DataFrame, pd.Series) else p
        positions = [strat(*[s] + args) for strat, args in zip(strats, inputs)]
        positions = trim_positions(positions)
        #unique_positions = np.unique(positions, axis=0, return_index=True)  # remove duplicate position signals
        positions = positions
        s_cut = np.array(s[-positions.shape[1]:])
        returns += returns_from_positions(s_cut, positions)
    returns = trim_positions(returns)
    if keep_index:
        returns = pd.DataFrame(np.transpose(returns),
                               index=s.index[-returns.shape[1]:])
    else:
        returns = pd.DataFrame(np.transpose(returns))
    return returns


def single_strat_returns(inputs, price_data, strat):
    return build_returns(inputs, 1, price_data, [strat])


def optimize_portfolio(price_data, strats, inputs, opt_type='min_volatility',
                       returns_type='mean_historical_return', ):
    if type(price_data) == pd.core.series.Series:
        n_s = 1
    elif type(price_data) == pd.core.frame.DataFrame:
        n_s = len(price_data.columns)
    elif type(price_data) == np.ndarray:
        n_s = price_data.shape[0]
    else:
        raise Exception("Price data must be either x pandas data frame, pandas data series, or numpy array")

    assert sum([strat_input_lengths[strat] for strat in strats]) * n_s == len(inputs), \
        f"Input length ({len(inputs)}) must match length of strat inputs ({sum([strat_input_lengths[strat] for strat in strats]) * n_s})."

    returns = build_returns(inputs, n_s, price_data, strats)

    mu = getattr(expected_returns, returns_type)(returns, returns_data=True)
    S = risk_models.sample_cov(returns, returns_data=True)

    ef = EfficientFrontier(mu, S)
    #ef.add_objective(objective_functions.L2_reg, gamma=0.1)
    weights = getattr(ef, opt_type)()

    return ef.portfolio_performance(), returns, weights


def test_portfolio(price_data, strats, inputs, weights, test_start_date, test_end_date):
    if type(price_data) == pd.core.series.Series:
        n_s = 1
        price_data = price_data.to_frame()
    elif type(price_data) == pd.core.frame.DataFrame:
        n_s = len(price_data.columns)
    else:
        raise Exception("Price data must be either x pandas data frame or data series")

    assert sum([strat_input_lengths[strat] for strat in strats]) * n_s == len(inputs), \
        f"Input length ({len(inputs)}) must match length of strat inputs ({sum([strat_input_lengths[strat] for strat in strats]) * n_s})."

    returns = build_returns(inputs, n_s, price_data, strats, keep_index=True)[test_start_date:test_end_date]
    mu = expected_returns.mean_historical_return(returns, returns_data=True)
    S = risk_models.CovarianceShrinkage(returns, returns_data=True).ledoit_wolf()

    perf = base_optimizer.portfolio_performance(
        weights, mu, S, verbose=True, risk_free_rate=0.02
    )

    return perf


def get_strat_names(strats, inputs, symbols):
    names = []
    n_s = len(symbols)
    inputs = split_inputs(strats, inputs, n_s)

    for (j, symb) in enumerate(symbols):
        for (i, strat) in enumerate(strats):
            strat_func_name = str(strat).split()[1]
            if strat_input_lengths[strat] > 0:
                strat_inputs = inputs[i+j]
            else:
                strat_inputs = ''
            names += [f'{symb} - {strat_func_name}{strat_inputs}']

    return names


def get_portfolio_returns_series(price_data, strats, inputs, weights):
    if type(price_data) == pd.core.series.Series:
        n_s = 1
        price_data.to_frame()
    elif type(price_data) == pd.core.frame.DataFrame:
        n_s = len(price_data.columns)
    else:
        raise Exception("Price data must be either x pandas data frame or data series")

    strat_returns = build_returns(inputs, n_s, price_data, strats)
    total_returns = np.array(strat_returns) @ weights

    return pd.Series(total_returns, index=strat_returns.index)


def optimize_system(optimizer, func, x0, **kwargs):
    return optimizer(func, x0, **kwargs)


def pygad_optimizer(func, x0, num_generations=100, num_parents_mating=4, sol_per_pop=8, init_range_low=0,
                    init_range_high=200, parent_selection_type="sss", keep_parents=1, crossover_type="single_point",
                    mutation_type="random", mutation_percent_genes=10):
    num_genes = len(x0)

    ga_instance = pygad.GA(num_generations=num_generations,
                           num_parents_mating=num_parents_mating,
                           fitness_func=func,
                           sol_per_pop=sol_per_pop,
                           num_genes=num_genes,
                           init_range_low=init_range_low,
                           init_range_high=init_range_high,
                           parent_selection_type=parent_selection_type,
                           keep_parents=keep_parents,
                           crossover_type=crossover_type,
                           mutation_type=mutation_type,
                           mutation_percent_genes=mutation_percent_genes,
                           )

    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    return solution, solution_fitness


def main():
    symbols = ['GOOG', 'SPY', 'TSLA', 'AMZN', 'FB']
    returns_type = 'mean_historical_return'
    opt_type = 'max_sharpe'

    train_start_date = '2018-01-01'
    train_end_date = '2020-01-01'

    test_start_date = train_end_date
    test_end_date = '2020-02-01'

    price_data = get_data(symbols, start_date=train_start_date, end_date=test_end_date)
    train_data = price_data[train_start_date:train_end_date]

    if type(train_data) == pd.core.series.Series:
        n_s = 1
    elif type(train_data) == pd.core.frame.DataFrame:
        n_s = len(train_data.columns)

    n_ma_strats = 1
    fast_window_sizes = [randint(10, 50) for i in range(n_ma_strats * 2 * n_s)]
    x0 = [(fw, fw + randint(50, 200)) for fw in fast_window_sizes]

    strats = [moving_average_crossover_strat] * n_ma_strats + [buy_and_hold_strat]
    train_data_np = np.transpose(np.array(train_data))

    def min_volatility_fitness_func(solution, solution_idx):
        try:
            return 1 / optimize_portfolio(train_data_np, strats, solution, returns_type=returns_type)[0][1]
        except (ArpackNoConvergence, OptimizationError):
            return 0

    def max_sharpe_fitness_func(solution, solution_idx):
        try:
            return optimize_portfolio(train_data_np, strats, solution, opt_type=opt_type,
                                      returns_type=returns_type)[0][2]
        except (ArpackNoConvergence, OptimizationError):
            return 0

    def neg_sharpe_func(solution, solution_idx):
        try:
            return -optimize_portfolio(train_data_np, strats, solution, opt_type=opt_type,
                                       returns_type=returns_type)[0][2]
        except (ArpackNoConvergence, OptimizationError):
            return 0

    objective_function = max_sharpe_fitness_func if opt_type == 'max_sharpe' else min_volatility_fitness_func

    solution, solution_fitness = optimize_system(pygad_optimizer, objective_function, x0, num_generations=50,
                                                 init_range_low=1, init_range_high=200)

    r = test_system(opt_type, returns_type, solution, strats, price_data,
                    (train_start_date, train_end_date, test_start_date, test_end_date))

    print(f'performance ration: {r["system"][2]/r["baseline"][2]}')

    weights = r['strat_weights']
    print(solution, weights)
    test_data = price_data[train_start_date:test_end_date]
    strat_names = np.array(get_strat_names(strats, solution, symbols))

    plt.bar(strat_names[weights > 0], weights[weights > 0])
    plt.show()

    portfolio_returns = get_portfolio_returns_series(test_data, strats, solution, weights)
    portfolio_returns = portfolio_returns
    portfolio_returns.plot()
    plt.show()


def test_system(opt_type, returns_type, solution, strats, price_data, dates):
    (train_start_date, train_end_date, test_start_date, test_end_date) = dates
    train_data = price_data[train_start_date:train_end_date]
    mu = getattr(expected_returns, returns_type)(train_data)
    S = risk_models.CovarianceShrinkage(train_data).ledoit_wolf()
    ef = EfficientFrontier(mu, S)
    baseline_weights = getattr(ef, opt_type)()
    baseline_train_results = ef.portfolio_performance()
    baseline_test_data = price_data[test_start_date:test_end_date]
    mu = expected_returns.mean_historical_return(baseline_test_data)
    S = risk_models.sample_cov(baseline_test_data)
    print("Baseline Test Results:")
    baseline_test_results = base_optimizer.portfolio_performance(
        baseline_weights, mu, S, verbose=True, risk_free_rate=0.02
    )
    results = optimize_portfolio(train_data, strats, solution, opt_type=opt_type,
                                 returns_type=returns_type)
    print(results[0])
    weights = np.array(list(results[2].values()))
    test_data = price_data[train_start_date:test_end_date]
    print("Test Results:")
    test_results = test_portfolio(test_data, strats, solution, weights, test_start_date, test_end_date)
    return {'baseline': baseline_test_results, 'system': test_results, 'strat_weights': weights}


def main2():
    train_start_date = '2018-01-01'
    train_end_date = '2020-01-01'

    test_start_date = train_end_date
    test_end_date = '2021-01-01'

    symbols = ['FB']

    strat = moving_average_crossover_strat

    price_data = get_data(symbols, start_date=train_start_date, end_date=test_end_date)
    train_data = price_data[train_start_date:train_end_date]
    test_data = price_data[train_start_date:test_end_date]

    if type(train_data) == pd.core.series.Series:
        n_s = 1
    elif type(train_data) == pd.core.frame.DataFrame:
        n_s = len(train_data.columns)

    bounds = [[1, 200], [1, 200]]

    def objective_function(x):
        returns = single_strat_returns(x, train_data, strat)
        mu = expected_returns.ema_historical_return(returns, returns_data=True)
        S = risk_models.CovarianceShrinkage(returns, returns_data=True).ledoit_wolf()

        perf = base_optimizer.portfolio_performance(
            [1], mu, S, verbose=False, risk_free_rate=0.02
        )
        return -perf[2]  # negative sharpe ratio

    results = dict()
    #results['shgo'] = optimize.shgo(objective_function, bounds)
    #results['DA'] = optimize.dual_annealing(objective_function, bounds)
    results['DE'] = optimize.differential_evolution(objective_function, bounds)
    #results['BH'] = optimize.basinhopping(objective_function, bounds)

    results = results['DE']

    opt_sharpe = -results['fun']
    x = results['x']

    print(opt_sharpe)

    test_portfolio(test_data, [strat], x, [1], test_start_date, test_end_date)


if __name__ == "__main__":
    main()
