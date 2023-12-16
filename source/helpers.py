from scipy.stats import friedmanchisquare
import numpy as np
import json
import os


def table_header():
    result = "| {:<19} | ".format("Function/Rank")
    functions = ["DE Rand 1", "DE Best 1", "PSO", "SOMA all-to-one", "SOMA all-to-all"]
    formatted_funcs = ["{:^17}".format(func) for func in functions]
    result += " | ".join(formatted_funcs)
    result += " | "
    result += "{:^10}".format("Avg. Diff.")
    result += " |"
    table_separator()
    print(result)
    table_separator()


def table_footer(average_ranks, average_distance):
    result = "| {:<19} | ".format("Average")
    formatted_ranks = ["{:^17}".format(rank) for rank in average_ranks]
    result += " | ".join(formatted_ranks)
    result += " | "
    result += "{:^10.4f}".format(average_distance)
    result += " |"
    table_separator()
    print(result)
    table_separator()


def table_row(name, array, distance):
    result = "| {:<19} | ".format(name)
    formatted_ranks = ["{:^17}".format(rank) for rank in array]
    result += " | ".join(formatted_ranks)
    result += " | "
    result += "{:^10.4f}".format(distance)
    result += " |"
    print(result)


def table_separator():
    print("-" * 136)


def euclidean_distance(vector1, vector2):
    return np.linalg.norm(np.array(vector1) - np.array(vector2))


# saving results (2D array) to a file
def save_results(results, filename):
    with open(filename, "w") as f:
        json.dump(results, f, indent=4)


def load_results(filename):
    results = {}

    if os.path.exists(filename):
        with open(filename, "r") as f:
            results = json.load(f)

    return results


def friedman_test(dimension_ranks):
    return friedmanchisquare(*dimension_ranks)


def rank_array(arr):
    sorted_arr = sorted(arr)
    rank_dict = {value: rank for rank, value in enumerate(sorted_arr, start=1)}
    ranked_array = [rank_dict[value] for value in arr]
    return ranked_array


def printable_ranked_array(arr):
    ranked = rank_array(arr)
    formatted_ranks = ["{:>17}".format(rank) for rank in ranked]
    result = " | ".join(formatted_ranks)
    return result


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


def bounce_vector(vector, bounds):
    """
    Calculate the bounce of a vector within specified bounds.

    Parameters:
    - vector (list): The input vector.
    - bounds (list of tuples): The lower and upper bounds for each component of the vector.

    Returns:
    - list: The bounced vector.
    """

    lower_bounds, upper_bounds = bounds[:, 0], bounds[:, 1]

    # Bounce if the value exceeds the bounds
    bounced_vector = np.where(vector < lower_bounds, lower_bounds, vector)
    bounced_vector = np.where(vector > upper_bounds, upper_bounds, bounced_vector)

    return bounced_vector


class Particle:
    def __init__(self, num_dimensions, bounds):
        self.position = np.random.uniform(
            bounds[:, 0], bounds[:, 1], size=num_dimensions
        )
        self.velocity = np.random.uniform(-1, 1, size=num_dimensions)
        self.best_position = self.position.copy()
        self.best_value = float("inf")
