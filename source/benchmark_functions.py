import numpy as np
import source.helpers as hp
import json

__name__ = "benchmark_functions"

# Load function's data from json file
with open("./source/data.json", "r") as file:
    data = json.load(file)


@hp.set_function_info(
    name="De Jong N1",
    source="MatInf presentation",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} x_i^2$",
)
def dejong1(x):
    return sum(xi**2 for xi in x)


@hp.set_function_info(
    name="Schwefel's function",
    source="MatInf presentation",
    formula="$f(\\mathbf{x}) = 418.9829n - \\sum_{i=1}^{n} x_i \\sin(\\sqrt{|x_i|})$",
)
def schwefel(x):
    result = 418.9829 * len(x) - sum(xi * np.sin(np.sqrt(abs(xi))) for xi in x)
    return result


@hp.set_function_info(
    name="Ackley's 1st function",
    source="M. Jamil et al.",
    formula="$f(\\mathbf{x}) = -20 \\exp\\left(-0.2 \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} x_i^2}\\right) - \\exp\\left(\\frac{1}{n} \\sum_{i=1}^{n} \\cos(2\\pi x_i)\\right) + 20 + e$",
)
def ackley(x):
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(2 * np.pi * xi) for xi in x)
    result = -20 * np.exp(-0.2 * np.sqrt(sum1 / n)) - np.exp(sum2 / n) + 20 + np.e
    return result


@hp.set_function_info(
    name="Alpine 1st function",
    source="M. Jamil et al.",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} |x_i \\sin(x_i) + 0.1x_i|$",
)
def alpine1(x):
    result = sum(abs(xi * np.sin(xi) + 0.1 * xi) for xi in x)
    return result


# Jamil
@hp.set_function_info(
    name="Alpine 2nd function",
    source="M. Jamil et al.",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} \\sin(x_i) \\sin(\\sqrt{|x_i|})$",
)
def alpine2(x):
    result = sum(np.sin(xi) * np.sin(np.sqrt(abs(xi))) for xi in x)
    return result


# # Jamil
# @hp.set_function_info(name="Brown function", source="M. Jamil et al.")
# def brown(v):
#     hp.normalize_function_input(v, -500, 500)
#     result = 0
#     for i in range(len(v) - 1):
#         result += (v[i] ** 2) ** (v[i + 1] ** 2 + 1) + (v[i + 1] ** 2) ** (
#             v[i] ** 2 + 1
#         )
#     return result


# # Jamil - very simple
# @hp.set_function_info(name="Chung Reynolds function", source="M. Jamil et al.")
# def chung_reynolds(v):
#     hp.normalize_function_input(v, -500, 500)
#     result = 0
#     for i in range(len(v) - 1):
#         result += i**2
#     return result**2


# Jamil
@hp.set_function_info(
    name="Csendes function",
    source="M. Jamil et al.",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} (x_i^6(2 + \\sin(1/x_i)))$",
)
def csendes(x):
    result = sum(xi**6 * (2 + np.sin(1 / xi)) for xi in x)
    return result


# # copilot
# @hp.set_function_info(name="Exponential function", source="Github copilot autocomplete")
# def exponential(v):
#     hp.normalize_function_input(v, -500, 500)
#     result = 0
#     for i in range(len(v)):
#         result += np.exp(-2 * np.log(2) * ((v[i] - 0.1) / 0.8) ** 2) * (
#             np.sin(5 * np.pi * v[i]) ** 6
#         )
#     return -result


# # Jamil - [-1, 1]
# @hp.set_function_info(name="Exponential function v2", source="M. Jamil et al.")
# def exponential2(v):
#     hp.normalize_function_input(v, -1, 1)
#     result = 0
#     for i in range(len(v)):
#         result += i**2
#     result *= -0.5
#     return np.exp(result) * -1


# Jamil - [-100, 100]
@hp.set_function_info(
    name="Griewank",
    source="M. Jamil et al.",
    formula="$f(\\mathbf{x}) = 1 + \\frac{1}{4000} \\sum_{i=1}^{n} x_i^2 - \\prod_{i=1}^{n} \\cos(\\frac{x_i}{\\sqrt{i}})$",
)
def griewank(x):
    sum_part = sum(xi**2 / 4000 for xi in x)
    prod_part = np.prod([np.cos(xi / np.sqrt(i + 1)) for i, xi in enumerate(x)])
    result = 1 + sum_part - prod_part
    return result


# # Jamil
# @hp.set_function_info(name="Pathological function", source="M. Jamil et al.")
# def pathological(v):
#     hp.normalize_function_input(v, -100, 100)
#     result = 0
#     for i in range(len(v)):
#         result += (
#             0.5
#             + (np.sin(np.sqrt(100 * v[i] ** 2))) ** 2
#             - 0.5 / (1 + 0.001 * v[i] ** 2) ** 2
#         )
#     return result


# Jamil
@hp.set_function_info(
    name="Quartic",
    source="M. Jamil et al.",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} i x_i^4$",
)
def quartic(x):
    result = sum((i + 1) * xi**4 for i, xi in enumerate(x))
    return result


# # Jamil
# @hp.set_function_info(name="Rosenbrock function", source="M. Jamil et al.")
# def rosenbrock(v):
#     hp.normalize_function_input(v, -30, 30)
#     result = 0
#     for i in range(len(v) - 1):
#         result += 100 * (v[i + 1] - v[i] ** 2) ** 2 + (v[i] - 1) ** 2
#     return result


# TODO: Solve 2D
# # Jamil
# @hp.set_function_info(
#     name="Schaffer f6 function",
#     source="M. Jamil et al.",
#     formula="$f(\\mathbf{x}) = 0.5 + \\frac{\\sin^2(\\sqrt{x_1^2 + x_2^2}) - 0.5}{[1 + 0.001(x_1^2 + x_2^2)]^2}$",
# )
# def schaffer_f6(x):
#     if len(x) != 2:
#         raise ValueError("Schaffer F6 function is defined for n = 2")
#     x1, x2 = x
#     result = (
#         0.5
#         + (np.sin(np.sqrt(x1**2 + x2**2)) ** 2 - 0.5)
#         / (1 + 0.001 * (x1**2 + x2**2)) ** 2
#     )
#     return result


# Jamil
@hp.set_function_info(
    name="Schwefel 2.22",
    source="M. Jamil et al.",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} |x_i| + \\prod_{i=1}^{n} |x_i|$",
)
def schwefel_222(x):
    result = sum([abs(xi) for xi in x]) + np.prod([abs(xi) for xi in x])
    return result


# # Jamil
# @hp.set_function_info(name="Weirerstrass function", source="M. Jamil et al.")
# def weirerstrass(v):
#     hp.normalize_function_input(v, -0.5, 0.5)
#     result = 0
#     for i in range(len(v)):
#         for j in range(21):
#             result += 0.5**j * np.cos(2 * np.pi * 3**j * (v[i] + 0.5))
#     result2 = 0
#     for j in range(21):
#         result2 += 0.5**j * np.cos(2 * np.pi * 3**j * 0.5)
#     return result - len(v) * result2


@hp.set_function_info(
    name="Levy",
    source="[1] - LLM's unmodified version",
    formula="$f(\\mathbf{x}) = \\sin^2(\\pi w_1) + \\sum_{i=1}^{n-1} (w_i - 1)^2 [1 + 10 \\sin^2(\\pi w_i + 1)] + (w_n - 1)^2 [1 + \\sin^2(2\\pi w_n)]$",
)
def levy(x):
    n = len(x)
    w = [1 + (xi - 1) / 4 for xi in x]

    term1 = np.sin(np.pi * w[0]) ** 2
    term2 = sum(
        (w[i] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[i] + 1) ** 2) for i in range(n - 1)
    )
    term3 = (w[n - 1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[n - 1]) ** 2)

    result = term1 + term2 + term3
    return result
