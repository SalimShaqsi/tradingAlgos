import inspect

sig = inspect.signature


def get_func_arg_names(func):
    return [p for p, d in sig(func).parameters.items() if d.default == inspect._empty and p not in ['args', 'kwargs']]


if __name__ == '__main__':

    def func(x, *args, y=2, **kwargs):
        return x + y

    print(get_func_arg_names(func))
