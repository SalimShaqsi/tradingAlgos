import numpy as np
from ortools.linear_solver import pywraplp


def pre_calc(s, positions):
    s = np.array(s)
    price_ratios = s[1:] / s[:-1]
    n = len(price_ratios)
    returns = price_ratios - 1
    ln_price_ratios = np.log(price_ratios)
    # total compounded return for each strat / stock pair
    strat_total_returns = np.exp(np.matmul(positions[:, :-1], ln_price_ratios)) - 1
    strat_returns = returns * positions[:, :-1]
    strat_total_returns = np.mean(strat_returns, axis=1) * 252
    covariance_matrix = np.cov(strat_returns) * 252

    return {'strat_total_returns': strat_total_returns, 'covariance_matrix': covariance_matrix}


def simple_solve(s, positions, var_limit=None, warm_weights=None, precalc_data=None):
    n_t = len(s) - 1
    n_i = positions.shape[0]

    assert n_t == positions.shape[1] - 1, \
        "The length of the each position array must equal the length of the prices array"

    precalc_data = precalc_data or pre_calc(s, positions)

    strat_total_returns = precalc_data['strat_total_returns']
    covariance_matrix = precalc_data['covariance_matrix']

    if not var_limit:
        max_index = np.argmax(strat_total_returns)

        total_return = strat_total_returns[max_index]

        x_ = np.zeros(n_i)

        x_[max_index] = 1
        return {
            'status_code': '0' if total_return > 0 else 2,
            'strat_weights': x_,
            'total_return': max(0, total_return),
            'return_variance': covariance_matrix[max_index, max_index] if total_return > 0 else 0
        }

    solver = pywraplp.Solver.CreateSolver('SCIP')

    infinity = solver.infinity()

    x = [solver.NumVar(0, infinity, f'x_{i}') for i in range(n_i)]

    gain_expression = solver.Sum([x[i] * strat_total_returns[i] for i in range(n_i)])

    solver.Add(solver.Sum(x) <= 1)

    if warm_weights is None:
        warm_weights = np.full(n_i, 1)

    variance_expression = solver.Sum([
        solver.Sum([
            x[i] * warm_weights[j] * covariance_matrix[i, j] for i in range(n_i)
        ]) for j in range(n_i)
    ])

    solver.Add(variance_expression <= var_limit)

    solver.Add(solver.Sum(x) >= 1)

    solver.Maximize(gain_expression)

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL and solver.Objective().Value() <= 0:
        x_ = np.zeros(n_i)
        var = total_return = 0
        status_code = '2'
    elif status == pywraplp.Solver.OPTIMAL:
        x_ = np.array([var.solution_value() for var in x])
        if sum(x_) < 0.999:
            status_code = '1'

        else:
            status_code = '0'
        total_return = np.dot(strat_total_returns, x_)
        var = x_ @ covariance_matrix @ x_
    else:
        x_ = np.zeros(n_i)
        var = total_return = 0
        status_code = '3'

    return {
        'status_code': status_code,
        'strat_weights': x_,
        'total_return': total_return,
        'return_variance': var
    }


def dual_sovle(s, positions, var_limit, warm_weights, precalc_data=None):
    n_t = len(s) - 1
    n_i = positions.shape[0]

    assert n_t == positions.shape[1] - 1, \
        "The length of the each position array must equal the length of the prices array"

    precalc_data = precalc_data or pre_calc(s, positions)

    strat_total_returns = precalc_data['strat_total_returns']
    covariance_matrix = precalc_data['covariance_matrix']

    solver = pywraplp.Solver.CreateSolver('SCIP')

    infinity = solver.infinity()

    y = [solver.NumVar(0, infinity, f'y_{i}') for i in range(2)]

    for i in range(n_i):
        solver.Add(y[0] * np.dot(warm_weights, covariance_matrix[i]) - y[1] >= strat_total_returns[i])
    solver.Minimize(var_limit * y[0] - y[1])

    status = solver.Solve()

    if status == pywraplp.Solver.OPTIMAL:
        return solver.Objective().Value(), np.array([var.solution_value() for var in y])
    else:
        print("no optimal solution")


def min_variance(s, positions, precalc_data=None):
    n_t = len(s) - 1
    n_i = positions.shape[0]

    assert n_t == positions.shape[1] - 1, \
        "The length of the each position array must equal the length of the prices array"

    precalc_data = precalc_data or pre_calc(s, positions)

    strat_total_returns = precalc_data['strat_total_returns']
    covariance_matrix = precalc_data['covariance_matrix']

    solve_matrix = np.vstack((2*covariance_matrix, np.ones(n_i)))

    solve_matrix = np.insert(solve_matrix, n_i, np.ones(n_i + 1), axis=1)

    solve_matrix[-1, -1] = 0

    b = np.zeros(n_i + 1)
    b[-1] = 1

    x_ = np.linalg.solve(solve_matrix, b)[:-1]

    total_return = np.dot(x_, strat_total_returns)

    return_variance = x_ @ covariance_matrix @ x_

    return {'strat_weights': x_, 'total_return': total_return, 'return_variance': return_variance}


def enumerate(s, positions, precalc_data=None):
    n_t = len(s) - 1
    n_i = positions.shape[0]

    assert n_t == positions.shape[1] - 1, \
        "The length of the each position array must equal the length of the prices array"

    assert n_i <= 2, "Brute force method only supports up to two variables"

    precalc_data = precalc_data or pre_calc(s, positions)

    strat_total_returns = precalc_data['strat_total_returns']
    covariance_matrix = precalc_data['covariance_matrix']

    x1 = np.linspace(0, 1, 10_000)
    x2 = 1 - x1

    x = np.vstack((x1, x2))

    returns = np.dot(np.transpose(x), strat_total_returns)
    vars = np.array([p @covariance_matrix @ p for p in np.transpose(x)])

    return np.transpose(np.vstack((returns, vars)))


# Faster than is_pareto_efficient_simple, but less readable.
def is_pareto_efficient(costs, return_mask=True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
