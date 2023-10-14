def normalize_function_input(v, lower_bound, upper_bound):
    result = []
    for i in range(len(v)):
        result.append((v[i] - lower_bound) / (upper_bound - lower_bound))
    return result