import numpy as np
import source.helpers as hp

__name__ = "benchmark_functions"


@hp.set_function_info(
    name="De Jong 1",
    source="MatInf presentation",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} x_i^2$",
)
def dejong1(x):
    hp.normalize_function_input(x, -500, 500)
    return sum(xi**2 for xi in x)


@hp.set_function_info(
    name="De Jong 2",
    source="MatInf presentation",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} (x_i^2 - 10\\cos(2\\pi x_i) + 10)$",
)
def dejong2(x):
    n = len(x)
    return sum(xi**2 - 10 * np.cos(2 * np.pi * xi) + 10 for xi in x)


# # MatInf presentation
# @hp.set_function_info(name="Schwefel's function", source="MatInf presentation")
# def schwefel(v):
#     hp.normalize_function_input(v, -500, 500)
#     result = 0
#     for i in range(len(v)):
#         result += -v[i] * np.sin(np.sqrt(np.abs(v[i])))
#     return result


# # Jamil
# @hp.set_function_info(name="Ackley's 1st function", source="M. Jamil et al.")
# def ackley1(v):
#     hp.normalize_function_input(v, -500, 500)
#     result = 0
#     for i in range(len(v)):
#         result += v[i] ** 2
#     result = -20 * np.exp(-0.2 * np.sqrt(result / len(v)))

#     sum_of_cosines = 0

#     for i in range(len(v)):
#         v[i] *= 2 * np.pi
#         sum_of_cosines += np.cos(v[i])

#     # Calculate the second part of the result
#     part2 = 20 - np.exp(sum_of_cosines / len(v))

#     result += part2
#     return result


# # Jamil
# @hp.set_function_info(name="Alpine 1st function", source="M. Jamil et al.")
# def alpine1(v):
#     hp.normalize_function_input(v, -500, 500)
#     result = 0
#     for i in range(len(v)):
#         result += np.abs(v[i] * np.sin(v[i]) + 0.1 * v[i])
#     return result


# # Jamil
# @hp.set_function_info(name="Alpine 2nd function", source="M. Jamil et al.")
# def alpine2(v):
#     hp.normalize_function_input(v, -500, 500)
#     result = 0
#     for i in range(len(v)):
#         result *= np.sqrt(v[i]) * np.sin(v[i])
#     return result


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


# # Jamil
# @hp.set_function_info(name="Csendes function", source="M. Jamil et al.")
# def csendes(v):
#     hp.normalize_function_input(v, -500, 500)
#     result = 0
#     for i in range(len(v)):
#         result += (v[i] ** 6) * (2 + np.sin(1 / v[i]))
#     return result


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


# # Jamil - [-100, 100]
# @hp.set_function_info(name="Griewank function", source="M. Jamil et al.")
# def griewank(v):
#     hp.normalize_function_input(v, -100, 100)
#     result = 0
#     for i in range(len(v)):
#         result += v[i] ** 2
#     result /= 4000
#     result2 = 1
#     for i in range(len(v)):
#         result2 *= np.cos(v[i] / np.sqrt(i + 1))
#     return result - result2 + 1


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


# # Jamil
# @hp.set_function_info(name="Quartic function", source="M. Jamil et al.")
# def quartic(v):
#     hp.normalize_function_input(v, -1.28, 1.28)
#     result = 0
#     for i in range(len(v)):
#         result += i * v[i] ** 4 + np.random.uniform(0, 1)
#     return result


# # Jamil
# @hp.set_function_info(name="Rosenbrock function", source="M. Jamil et al.")
# def rosenbrock(v):
#     hp.normalize_function_input(v, -30, 30)
#     result = 0
#     for i in range(len(v) - 1):
#         result += 100 * (v[i + 1] - v[i] ** 2) ** 2 + (v[i] - 1) ** 2
#     return result


# # Jamil
# @hp.set_function_info(name="Schaffer f6 function", source="M. Jamil et al.")
# def schaffer_f6(v):
#     hp.normalize_function_input(v, -100, 100)
#     result = 0
#     for i in range(len(v) - 1):
#         result += (
#             0.5
#             + ((np.sin(v[i] ** 2 - v[i + 1] ** 2)) ** 2 - 0.5)
#             / (1 + 0.001 * (v[i] ** 2 + v[i + 1] ** 2)) ** 2
#         )
#     return result


# # Jamil
# @hp.set_function_info(name="Trigonometric function", source="M. Jamil et al.")
# def trigonometric(v):
#     hp.normalize_function_input(v, -500, 500)
#     result = 0
#     result += 1
#     for i in range(len(v)):
#         result += (
#             8 * np.sin(7 * ((v[i] - 0.9) ** 2))
#             + 6 * np.sin(14 * (v[i] - 0.9))
#             + ((v[i] - 0.9) ** 2)
#         )
#     return result


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
