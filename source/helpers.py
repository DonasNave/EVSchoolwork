import numpy as np


def normalize_function_input(v, lower_bound, upper_bound):
    result = []
    for i in range(len(v)):
        result.append((v[i] - lower_bound) / 2 * (upper_bound - lower_bound))
    return result


def random_vector(dim, lower_bound, upper_bound):
    result = []
    for _ in range(dim):
        result.append(np.random.uniform(lower_bound, upper_bound))
    return result


def set_function_info(name, source, formula, bounds=(-100, 100)):
    def decorator(func):
        func._custom_name = name  # Set the custom name as an attribute
        func._source = source  # Set the source string as an attribute
        func._formula = formula  # Set the LaTeX formula as an attribute
        func._bounds = bounds
        return func

    return decorator
