# ---------------------------------------------------------------
# ---- All Copyright reserved (c) by David Parreño Barbuzano ----
# ---------------------------------------------------------------

import numpy as np

from solve_ffa import FireflyAlgorithm
from visualizer import VisualizerFFA2D

from utils import michalewicz, rosenbrock, de_yong, rastrigin, euclidean_distance
from utils import schwefel, ackley, easom, griewank, shubert
from utils import yang, sphere

# Examples of functions for fitness function
# Each key has a tuple value with the function
# and typical bounds for evaluation.
functions = {
    'michalewicz': (michalewicz, [0, np.pi]),
    'rosenbrock': (rosenbrock, [-5, 10]),
    'de_yong': (de_yong, [-5.12, 5.12]),
    'schwefel': (schwefel, [-500, 500]),
    'ackley': (ackley, [-32.768, 32.768]),
    'rastrigin': (rastrigin, [-5.12, 5.12]),
    'easom': (easom, [-100, 100]),
    'griewank': (griewank, [-600, 600]),
    'shubert': (shubert, [-10, 10]),
    'yang': (yang, [-10, 10]),
    'sphere': (sphere, [-5.12, 5.12])
}

# Selected fitness function for FFA
name_function = 'sphere'

# Hyperparameters for Firefly Algorithm
f_function, bounds = functions[name_function]
distance_function, mopt = euclidean_distance, 'm'
dim, max_gen, n_fireflies = 2, 10, 40
delta, alpha, beta = 0.97, 0.25, 1
beta_0, beta_min, gamma = 1, 0.25, 1
# bounds = [0, 4]

if __name__ == '__main__':
    ffa = FireflyAlgorithm(
        name='ffa', verbose=True, mopt=mopt, dim=dim, max_gen=max_gen,
        n_fireflies=n_fireflies, delta=delta, alpha=alpha, beta=beta,
        beta_0=beta_0, beta_min=beta_min, gamma=gamma, bounds=bounds,
        f_function=f_function, distance=distance_function
    )

    path_gif = 'ffa_' + name_function + '.gif'
    VisualizerFFA2D(solver=ffa, interval_time=500, file_path=path_gif).show()
