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
    bounds=(-5.12, 5.12),
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


# Jamil
@hp.set_function_info(
    name="Csendes function",
    source="M. Jamil et al.",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} (x_i^6(2 + \\sin(1/x_i)))$",
    bounds=(-1, 1),
)
def csendes(x):
    result = sum(xi**6 * (2 + np.sin(1 / xi)) for xi in x)
    return result


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


# Jamil
@hp.set_function_info(
    name="Quartic",
    source="M. Jamil et al.",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} i x_i^4$",
    bounds=(-1.28, 1.28),
)
def quartic(x):
    result = sum((i + 1) * xi**4 for i, xi in enumerate(x))
    return result


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


@hp.set_function_info(
    name="Rastrigin",
    source="[1] - LLM's unmodified version",
    formula="$f(\\mathbf{x}) = 10n + \\sum_{i=1}^{n} (x_i^2 - 10\\cos(2\\pi x_i))$",
    bounds=(-5.12, 5.12),
)
def rastrigin(x):
    n = len(x)
    result = 10 * n + sum(xi**2 - 10 * np.cos(2 * np.pi * xi) for xi in x)
    return result


@hp.set_function_info(
    name="Michalewicz",
    source="[1] - LLM's unmodified version",
    formula="$f(\\mathbf{x}) = -\\sum_{i=1}^{n} \\sin(x_i) \\sin^{2i}(x_i^2/\\pi)$",
    bounds=(0, np.pi),
)
def michalewicz(x):
    result = -sum(
        np.sin(xi) * np.sin((i + 1) * xi**2 / np.pi) ** 20 for i, xi in enumerate(x)
    )
    return result


@hp.set_function_info(
    name="Zakharov",
    source="[1] - LLM unmodified input, 2nd try",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} x_i^2 + \\left(\\sum_{i=1}^{n} 0.5i x_i\\right)^2 + \\left(\\sum_{i=1}^{n} 0.5i x_i\\right)^4$",
    bounds=(-10, 10),
)
def zakharov(x):
    term1 = sum(xi**2 for xi in x)
    term2 = sum((0.5 * (i + 1) * xi for i, xi in enumerate(x))) ** 2
    term3 = sum((0.5 * (i + 1) * xi for i, xi in enumerate(x))) ** 4
    result = term1 + term2 + term3
    return result


@hp.set_function_info(
    name="Styblinski-Tang",
    source="[1] - LLM's unmodified version",
    formula="$f(\\mathbf{x}) = \\frac{1}{2} \\sum_{i=1}^{n} (x_i^4 - 16x_i^2 + 5x_i)$",
    bounds=(-5, 5),
)
def styblinski_tang(x):
    result = 0.5 * sum(xi**4 - 16 * xi**2 + 5 * xi for xi in x)
    return result


@hp.set_function_info(
    name="Svanda 1st",
    source="Custom made",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} \\frac{x_i}{3} \\cos(x_i) + \\sum_{i=1}^{n} \\frac{x_i}{2} \\cos\\left(\\frac{2}{3}x_i\\right)$",
)
def svanda_1(x):
    result = sum(xi / 3 * np.cos(xi) for xi in x) + sum(
        xi / 2 * np.cos(xi * 2 / 3) for xi in x
    )
    return result


@hp.set_function_info(
    name="Svanda 2nd",
    source="Custom made",
    formula="$f(\\mathbf{x}) = -\\sum_{i=1}^{n} \\frac{1}{\\sqrt{|x_i|}} \\cos(x_i)$",
)
def svanda_2(x):
    result = -sum(1 / np.sqrt(np.abs(xi)) * np.cos(xi) for xi in x)
    return result


@hp.set_function_info(
    name="Svanda 3rd",
    source="Custom made",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} |x_i - 0.42 \\min(|x_j|)| \\cdot |x_i| \\cdot \\sin\\left(\\frac{x_i}{4}\\right)$",
)
def svanda_3(x):
    min_abs_x = min(abs(xi) for xi in x) * 0.42
    result = sum(abs(abs(xi) - min_abs_x) * abs(xi) * np.sin(xi / 4) for xi in x)
    return result


@hp.set_function_info(
    name="Svanda 4th",
    source="Custom made",
    formula="$f(\\mathbf{x}) = \\left|\\sum_{i=1}^{n} \\left(\\frac{1}{1 + e^{x_i}} + \\frac{x_i^2}{600}\\right)\\right| + \\frac{6}{0.5 + 0.1 \\cdot \\lVert \\mathbf{x} \\rVert}$",
)
def svanda_4(x):
    origin = np.zeros(len(x))
    distance = np.linalg.norm(x - origin)
    result = abs(sum(1 / (1 + np.exp(xi)) + (xi**2 / 600) for xi in x)) + 6 / (
        0.5 + distance * 0.1
    )
    return result


@hp.set_function_info(
    name="Svanda 5th",
    source="Custom made",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} \\sin(x_i) \\sqrt{|x_i|} x_i \\mod 4$",
)
def svanda_5(x):
    result = sum(np.sin(xi) * np.sqrt(np.abs(xi)) * (xi % 4) for xi in x)
    return result


@hp.set_function_info(
    name="Svanda 6th",
    source="Custom made",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} \\tan^2(\\frac{x_i}{8}) x_i \\mod 10$",
)
def svanda_6(x):
    result = -sum((np.tan(xi / 8) ** 2) * (xi % 10) for xi in x)
    return result


@hp.set_function_info(
    name="LLM - Customized 1",
    source="LLM prompt with context",
    formula="$f(\\mathbf{x}) = -\\sum_{i=1}^{n} \\frac{1}{\\sqrt{|x_i| + 1}} \\cos(2x_i)$",
)
def custom_1_llm(x):
    result = -sum(1 / np.sqrt(np.abs(xi) + 1) * np.cos(2 * xi) for xi in x)
    return result


@hp.set_function_info(
    name="LLM - Customized 2",
    source="LLM prompt with context",
    formula="$f(\\mathbf{x}) = \\left|\\sum_{i=1}^{n} \\left(\\frac{1}{1 + e^{2x_i}} + \\frac{x_i^3}{800}\\right)\\right| + \\frac{7}{0.6 + 0.15 \\cdot \\lVert \\mathbf{x} \\rVert}$",
)
def custom_2_llm(x):
    origin = np.zeros(len(x))
    distance = np.linalg.norm(x - origin)
    result = abs(sum(1 / (1 + np.exp(2 * xi)) + (xi**3 / 800) for xi in x)) + 7 / (
        0.6 + distance * 0.15
    )
    return result


@hp.set_function_info(
    name="LLM - Wavy Peaks",
    source="LLM prompt with context",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} \\sin(3x_i) + \\sum_{i=1}^{n} \\cos(2x_i)$",
)
def wavy_peaks_LLM(x):
    result = sum(np.sin(3 * xi) + np.cos(2 * xi) for xi in x)
    return result


@hp.set_function_info(
    name="Pronounced Twisted Valleys",
    source="Custom made",
    formula="$f(\\mathbf{x}) = \\sum_{i=1}^{n} |3x_i^3 - 4x_i^2\\sin(2x_i) + 2\\cos(3x_i)|$",
    bounds=(-5.12, 5.12),
)
def pronounced_twisted_valleys(x):
    result = sum(
        abs(3 * xi**3 - 4 * xi**2 * np.sin(2 * xi) + 2 * np.cos(3 * xi)) for xi in x
    )
    return result


@hp.set_function_info(
    name="Michalewicz Altered",
    source="Custom made",
    formula="$f(\\mathbf{x}) = -\\sum_{i=1}^{n} \\sin(x_i) \\sin((i + 1)|x_i| / \\sqrt{\\pi})^{20}$",
)
def michalewicz_altered(x):
    result = -sum(
        np.sin(xi) * np.sin((i + 1) * np.abs(xi) / np.sqrt(np.pi)) ** 20
        for i, xi in enumerate(x)
    )
    return result


@hp.set_function_info(
    name="Ackley Altered",
    source="Custom made",
    formula="$f(\\mathbf{x}) = -20 \\exp\\left(-0.2 \\sqrt{\\frac{1}{n} \\sum_{i=1}^{n} x_i^2}\\right) - \\exp\\left(\\frac{1}{n} \\sum_{i=1}^{n} \\cos(2\\pi x_i)\\right) + 20 + e$",
)
def ackley_altered(x):
    n = len(x)
    sum1 = sum(xi**2 for xi in x)
    sum2 = sum(np.cos(2 * np.pi * xi) for xi in x)
    result = -20 * np.exp(-0.2 * np.sqrt(sum1 / n) + np.exp(sum2 / n) + 20 + np.e)
    return result % 4
