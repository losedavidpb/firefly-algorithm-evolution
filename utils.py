# ---------------------------------------------------------------
# ---- All Copyright reserved (c) by David Parreño Barbuzano ----
# ---------------------------------------------------------------

import numpy as np

def euclidean_distance(pos_1, pos_2):
    """Returns euclidean distance between two positions. """
    result = [(x1 - x2)**2 for x1, x2 in zip(pos_1, pos_2)]
    return np.math.sqrt(sum(result))

def manhattan_distance(pos_1, pos_2):
    """Returns manhattan distance between two positions. """
    result = [(x1 - x2) for x1, x2 in zip(pos_1, pos_2)]
    return np.sum(result)

def michalewicz(p):
    """Returns Michalewicz's function evaluation. """
    m = 10
    return -np.sum([
        np.sin(p[i]) * (np.sin((i + 1) * p[i] ** 2 / np.pi)) ** (2 * m)
        for i in range(len(p))
    ])

def rosenbrock(p):
    """Returns Rosenbrock's function evaluation. """
    return np.sum([
        100 * (p[i + 1] - p[i]**2)**2 + (p[i] - 1)**2
        for i in range(0, len(p) - 1)
    ])

def de_yong(p):
    """Returns De Yong's function evaluation. """
    if len(p) == 2:
        a = np.zeros(shape=(2, 25))
        a[1] = np.repeat([-32, -16, 0, 16, 32], 5)

        for i in range(0, 25, 5):
            a[0][i:i+5] = [-32, -16, 0, 16, 32]

        aux = np.sum([
            1 / (i + (p[0] - a[0, i])**6 + (p[1] - a[1, i])**6)
            for i in range(len(p))
        ])
        return (0.002 + aux)**-1
    else:
        print("error de_yong: 'p' must be a 2-D vector")
        return float('-inf')

def schwefel(p):
    """Returns Schwefel's function evaluation. """
    aux = np.sum([
        p[i] * np.sin(np.sqrt(np.abs(p[i])))
        for i in range(len(p))
    ])
    return 418.9829 * len(p) - aux

def ackley(p):
    """Returns Ackley's function evaluation. """
    a, b, c, d = 20, 0.2, 2 * np.pi, len(p)
    result = -a * np.exp(-b * np.sqrt(1 / d * np.sum([p[i]**2 for i in range(d)])))
    result = result - np.exp(1 / d * np.sum([np.cos(c * p[i]) for i in range(d)]))
    return result + a + np.exp(1)

def rastrigin(p):
    """Returns Rastrigin's function evaluation. """
    return 10 * len(p) + np.sum([
        p[i]**2 - 10 * np.cos(2 * np.pi * p[i])
        for i in range(len(p))
    ])

def easom(p):
    """Returns Easom's function evaluation. """
    if len(p) == 2:
        result = -np.cos(p[0]) * np.cos(p[1])
        return result * np.exp(-(p[0] - np.pi)**2 - (p[1] - np.pi)**2)
    else:
        print("error easom: 'p' must be a 2-D vector")
        return float('-inf')

def griewank(p):
    """Returns Griewank's function evaluation. """
    result = np.sum([p[i]**2 / 4000 for i in range(len(p))])
    result = result - np.prod([np.cos(p[i - 1] / np.sqrt(i)) for i in range(1, len(p))]) + 1
    return result

def shubert(p):
    """Returns Shubert's function evaluation. """
    if len(p) == 2:
        result = np.sum([i * np.cos((i + 1) * p[0] + i) for i in range(5)])
        result *= np.sum([i * np.cos((i + 1) * p[1] + i) for i in range(5)])
        return result
    else:
        print("error shubert: 'p' must be a 2-D vector")
        return float('inf')

def yang(p):
    """Returns Yang's function evaluation. """
    m, a = 5, 15
    result = np.exp(-np.sum([(p[i] / a)**(2 * m) for i in range(len(p))]))
    result = result - 2 * np.exp(-np.sum(p**2))
    return result * np.prod([np.cos(p[i])**2 for i in range(len(p))])

def sphere(p):
    """Returns the Sphere function evaluation. """
    return np.sum(p**2)
