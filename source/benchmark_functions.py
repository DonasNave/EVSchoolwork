import numpy as np
import source.helpers as hp

# MatInf presentation
def dejong(v):
    hp.normalize_function_input(v, -500, 500)
    result = 0
    for i in range(len(v)):
        result += v[i] ** 2
    return result

# MatInf presentation
def dejong2(v):
    hp.normalize_function_input(v, -500, 500)
    result = 0
    for i in range(len(v) - 1):
        result += 100 * (v[i] ** 2 - v[i + 1]) ** 2 + (1 - v[i]) ** 2
    return result

# MatInf presentation
def schwefel(v):
    hp.normalize_function_input(v, -500, 500)
    result = 0
    for i in range(len(v)):
        result += -v[i] * np.sin(np.sqrt(np.abs(v[i])))
    return result

# Jamil
def ackley1(v):
    hp.normalize_function_input(v, -500, 500)
    result = 0
    for i in range(len(v)):
        result += v[i] ** 2
    result = -20 * np.exp(-0.2 * np.sqrt(result / len(v)))
    result += 20 - np.exp(np.sum(np.cos(2 * np.pi * v)) / len(v))
    return result

# Jamil
def alpine1(v):
    hp.normalize_function_input(v, -500, 500)
    result = 0
    for i in range(len(v)):
        result += np.abs(v[i] * np.sin(v[i]) + 0.1 * v[i])
    return result

# Jamil
def alpine2(v):
    hp.normalize_function_input(v, -500, 500)
    result = 0
    for i in range(len(v)):
        result *= np.sqrt(v[i]) * np.sin(v[i])
    return result

# Jamil
def brown(v):
    hp.normalize_function_input(v, -500, 500)
    result = 0
    for i in range(len(v) - 1):
        result += (v[i] ** 2) ** (v[i + 1] ** 2 + 1) + (v[i + 1] ** 2) ** (v[i] ** 2 + 1)
    return result

# Jamil - very simple
def chung_reynolds(v):
    hp.normalize_function_input(v, -500, 500)
    result = 0
    for i in range(len(v) - 1):
        result += i ** 2
    return result ** 2

# Jamil
def csendes(v):
    hp.normalize_function_input(v, -500, 500)
    result = 0
    for i in range(len(v)):
        result += (v[i] ** 6) * (2 + np.sin(1 / v[i]))
    return result

# copilot
def exponential(v):
    hp.normalize_function_input(v, -500, 500)
    result = 0
    for i in range(len(v)):
        result += np.exp(-2 * np.log(2) * ((v[i] - 0.1) / 0.8) ** 2) * (np.sin(5 * np.pi * v[i]) ** 6)
    return -result

# Jamil - [-1, 1]
def exponential2(v):
    hp.normalize_function_input(v, -1, 1)
    result = 0
    for i in range(len(v)):
        result += i ** 2
    result *= -0.5
    return np.exp(result) * -1

# Jamil - [-100, 100]
def griewank(v):
    hp.normalize_function_input(v, -100, 100)
    result = 0
    for i in range(len(v)):
        result += v[i] ** 2
    result /= 4000
    result2 = 1
    for i in range(len(v)):
        result2 *= np.cos(v[i] / np.sqrt(i + 1))
    return result - result2 + 1

# Jamil
def pathological(v):
    hp.normalize_function_input(v, -100, 100)
    result = 0
    for i in range(len(v)):
        result += 0.5 + (np.sin(np.sqrt(100 * v[i] ** 2))) ** 2 - 0.5 / (1 + 0.001 * v[i] ** 2) ** 2
    return result

# Jamil 
def quartic(v):
    hp.normalize_function_input(v, -1.28, 1.28)
    result = 0
    for i in range(len(v)):
        result += i * v[i] ** 4 + np.random.uniform(0, 1)
    return result

# Jamil
def rosenbrock(v):
    hp.normalize_function_input(v, -30, 30)
    result = 0
    for i in range(len(v) - 1):
        result += 100 * (v[i + 1] - v[i] ** 2) ** 2 + (v[i] - 1) ** 2
    return result

# Jamil
def schaffer_f6(v):
    hp.normalize_function_input(v, -100, 100)
    result = 0
    for i in range(len(v) - 1):
        result += 0.5 + ((np.sin(v[i] ** 2 - v[i + 1] ** 2)) ** 2 - 0.5) / (1 + 0.001 * (v[i] ** 2 + v[i + 1] ** 2)) ** 2
    return result

# Jamil
def trigonometric(v):
    hp.normalize_function_input(v, -500, 500)
    result = 0
    result += 1
    for i in range(len(v)):
        result += 8 * np.sin(7 * ((v[i] - 0.9) ** 2)) + 6 * np.sin(14 * (v[i] - 0.9)) + ((v[i] - 0.9) ** 2)
    return result

# Jamil
def weirerstrass(v):
    hp.normalize_function_input(v, -0.5, 0.5)
    result = 0
    for i in range(len(v)):
        for j in range(21):
            result += 0.5 ** j * np.cos(2 * np.pi * 3 ** j * (v[i] + 0.5))
    result2 = 0
    for j in range(21):
        result2 += 0.5 ** j * np.cos(2 * np.pi * 3 ** j * 0.5)
    return result - len(v) * result2