import numpy as np


def weight_function(x):
    return np.where(x < 0.5, 1, -2 * x + 2)


def calc(p):
    size = len(p)
    s1 = 0
    s2 = 0
    for i in range(size):
        x = i / (size - 1)
        w = weight_function(x)
        s1 += w * p[i]
        s2 += w
    return s1/s2
