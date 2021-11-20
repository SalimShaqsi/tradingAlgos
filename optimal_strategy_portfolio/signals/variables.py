from optimal_strategy_portfolio.signals import Signal


class Variable(Signal):
    is_array = False
    n_args = 1

    def __init__(self, name='variable', bounds=((-10_000, 10_000),), arg_types=(float,)):
        super().__init__(name=name, bounds=bounds, arg_types=arg_types)

    def transform(self, x, **kwargs):
        return x


class IntegerVariable(Variable):
    def __init__(self, name='integer_variable', bounds=((-10_000, 10_000),)):
        super().__init__(name=name, bounds=bounds, arg_types=(int,))


class BooleanVariable(Variable):
    def __init__(self, name='boolean_variable'):
        super().__init__(name=name, bounds=((0, 1),), arg_types=(int,))


Var = Variable
IntVar = IntegerVariable
BoolVar = BooleanVariable

