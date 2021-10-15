import functools
import numpy as np
from pypfopt.expected_returns import ema_historical_return
from pypfopt import risk_models, EfficientFrontier
from scipy import optimize
import pygad


def flip_objective(func):
    """Flips the sign of the objective function (for cases where a solver maximizes instead of minimizing)"""
    @functools.wraps(func)
    def wrapper_flip(*args, **kwargs):
        return -func(*args, **kwargs)
    return wrapper_flip


def scipy_optimize_wrapper(func, bounds, method='differential_evolution', **kwargs):
    r = getattr(optimize, method)(func, bounds, **kwargs)
    return r['fun'], r['x'], r


def print_generation_number(ga_instance):
    if ga_instance.generations_completed * 10 % ga_instance.num_generations == 0:
        print(f'Generations completed: {ga_instance.generations_completed} / {ga_instance.num_generations} '
              f'({ga_instance.generations_completed * 100 // ga_instance.num_generations}%)')


def pygad_wrapper(func, bounds, num_generations=200, num_parents_mating=4, sol_per_pop=8,
                  on_generation=print_generation_number, **kwargs):
    gene_space = [{'low': b[0], 'high': b[1]} for b in bounds]
    num_genes = len(gene_space)

    def fitness_func(solution, solution_idx):
        return -func(solution)  # '-' because pygad GA is a maximizer

    ga_instance = pygad.GA(fitness_func=fitness_func, gene_space=gene_space, num_generations=num_generations,
                           num_parents_mating=num_parents_mating, sol_per_pop=sol_per_pop, num_genes=num_genes,
                           on_generation=on_generation, **kwargs)
    ga_instance.run()
    solution, solution_fitness, solution_idx = ga_instance.best_solution()

    return solution_fitness, solution, solution_idx


def ledoit_wolf_covariance_shrinkage(prices, returns_data=False):
    return risk_models.CovarianceShrinkage(prices, returns_data=returns_data).ledoit_wolf()


def pypfopt_ef_wrapper(returns, expected_returns_model=ema_historical_return,
                       risk_model=ledoit_wolf_covariance_shrinkage, opt_type='max_sharpe', return_weights=False):
    mu = expected_returns_model(returns, returns_data=True, compounding=False)

    S = risk_model(returns, returns_data=True)

    ef = EfficientFrontier(mu, S)

    weights = getattr(ef, opt_type)()

    if return_weights:
        return weights

    if opt_type == 'max_sharpe':
        return -ef.portfolio_performance()[2]
    if opt_type == 'min_volatility':
        return ef.portfolio_performance()[1]
    raise NotImplementedError("Provided opt_type is not supported")





if __name__ == '__main__':
    def f(x):
        x, y = x[0], x[1]
        return x**2 + 4*y**2 + 2*x + 8*y

    bounds = np.array([[-10, 10], [-10, 10]])
    f_min, x_min, r = scipy_optimize_wrapper(f, bounds, method='dual_annealing')

    print(f_min, x_min)

    f_min, x_min, r = pygad_wrapper(f, bounds, num_generations=400, mutation_percent_genes=10, mutation_type="random")

    print(f_min, x_min)


