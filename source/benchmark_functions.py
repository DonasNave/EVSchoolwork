import numpy as np

def dejong(v):
    result = 0
    for i in range(len(v)):
        result += v[i] ** 2
    return result


def dejong2(v):
    result = 0
    for i in range(len(v) - 1):
        result += 100 * (v[i] ** 2 - v[i + 1]) ** 2 + (1 - v[i]) ** 2
    return result


def schwefel(v):
    result = 0
    for i in range(len(v)):
        result += -v[i] * np.sin(np.sqrt(np.abs(v[i])))
    return result