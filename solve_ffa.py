# ---------------------------------------------------------------
# ---- All Copyright reserved (c) by David Parreño Barbuzano ----
# ---------------------------------------------------------------

from distance_functions import euclidean_distance, manhattan_distance
from firefly import Firefly
from copy import deepcopy

import matplotlib.pyplot as plt
import math

# Just for debugging ...
debug_mode = True

class FireflyAlgorithm(object):

    def __init__(self, **kwargs):
        """Constructs a new instance of `FireflyAlgorithm` class.

        Valid parameters:
            * name              -- Firefly's Algorithm name (default: UNKNOWN)
            * verbose           -- Verbose mode for debugging (default: False)
            * dim               -- Dimension of the problem (default: 2)
            * bounds            -- Bounds of valid solutions (default: [0, 4])
            * max_gen           -- Max generations to execute (default: 10)
            * n_fireflies       -- Number of fireflies to be moved (default: 10)
            * f_function        -- Fitness function of the problem (default lambda **values: 0)
            * distance          -- Distance function (default: euclidean_distance)
            * delta             -- Delta coefficient to reduce randomness (default 0.97)
            * alpha             -- Randomness coefficient (default: 0.2)
            * beta              -- Attractiveness coefficient (default: 1)
            * beta_0            -- Initial attractiveness coefficient (default: 1)
            * beta_min          -- Min attractiveness coefficient (default 0.2)
            * gamma             -- Absorption light coefficient (default: 1)
            * mopt              -- Optimization method (default: m, valid: {M, m})

        """
        self.name = kwargs.get('name', 'UNKNOWN')
        self.verbose = kwargs.get('verbose', False)

        self.dim = kwargs.get('dim', 2)
        self.bounds = kwargs.get('bounds', [0, 4])
        self.max_gen = kwargs.get('max_gen', 10)
        self.n_fireflies = kwargs.get('n_fireflies', 10)

        self.f_function = kwargs.get('f_function', lambda **values: 0)
        self.distance = kwargs.get('distance', euclidean_distance)

        self.delta = kwargs.get('delta', 0.97)
        self.alpha = kwargs.get('alpha', 0.25)
        self.beta = kwargs.get('beta', 1)
        self.beta_0 = kwargs.get('beta_0', 1)
        self.beta_min = kwargs.get('beta_min', 0.2)
        self.gamma = kwargs.get('gamma', 0.97)

        self.mopt = kwargs.get('mopt', 'm')
        if self.mopt not in ['m', 'M']: self.mopt = 'm'

        opt_functions = {'m': lambda oc, nc: oc > nc, 'M': lambda oc, nc: oc < nc}
        self.optimize = opt_functions[self.mopt]

    def init_swarm(self):
        """Initialize random swarm population of fireflies. """
        swarm_population = []

        for i in range(self.n_fireflies):
            firefly = Firefly(dim=self.dim, bounds=self.bounds, alpha=self.alpha, beta=self.beta,
                              gamma=self.gamma, beta_min=self.beta_min, distance=self.distance, beta_0=self.beta_0)

            firefly.fitness_value = self.f_function(p=firefly.position)
            firefly.light = firefly.fitness_value

            swarm_population.append(firefly)

        return swarm_population

    def reduce_alpha(self):
        """Reduce randomness coefficient. """
        self.delta = 1 - (10**(-4) / 0.9) ** (1 / self.max_gen)
        self.alpha = (1 - self.delta) * self.alpha
        # self.alpha = self.alpha * self.delta

    def solve(self):
        """Implements firefly algorithm for current problem.

        Since intermediate solutions could be used for animations,
        this function would store swarm's history, so returned
        values are best cost, best firefly, and history.

        """
        history_fireflies = []
        reverse_val = self.mopt == 'm'

        swarm = self.init_swarm()
        swarm.sort(key=lambda f: f.light, reverse=reverse_val)

        best_cost = self.f_function(p=swarm[0].position)
        best_solution = deepcopy(swarm[0])

        if self.verbose is True:
            print(str.format(">> Evolution for FFA {} <<", self.name))

        for gen in range(self.max_gen):
            history_fireflies.append(deepcopy(swarm))

            if self.verbose is True:
                str_format = 'Generation {}: best_firefly={}, cost={}'
                print(str.format(str_format, gen, best_solution, best_cost))
                for firefly in swarm: print('\t *', firefly)

            for i in range(0, self.n_fireflies):
                for j in range(0, self.n_fireflies):
                    r = euclidean_distance(swarm[i].position, swarm[j].position)

                    if swarm[i].light_intensity(r) <= swarm[j].light_intensity(r):
                        swarm[i].move_towards(swarm[j])
                        swarm[i].fitness_value = self.f_function(p=swarm[i].position)

                swarm.sort(key=lambda f: f.fitness_value, reverse=reverse_val)
                swarm[0].random_move()

                if self.optimize(best_cost, swarm[0].fitness_value):
                    swarm[0].fitness_value = self.f_function(p=swarm[0].position)
                    best_cost, best_solution = swarm[0].fitness_value, deepcopy(swarm[0])

            self.reduce_alpha()
            for i in range(len(swarm)): swarm[i].alpha = self.alpha

        return best_cost, best_solution, history_fireflies

# -------------------------------------------------------------
# --------- Test for simple Firefly Algorithm -----------------
# -------------------------------------------------------------

if __name__ == '__main__':
    if debug_mode:
        def draw_plot(bounds, history):
            def get_position(h, num_dim):
                return [h[k].position[num_dim] for k in range(len(h))]

            def show_current_plot(color):
                plt.title('Evolution of Firefly Algorithm')
                plt.suptitle('Generation ' + str(i)), plt.grid()
                plt.plot(x, y, color + '.')
                plt.axis([bounds[0], bounds[1], bounds[0], bounds[1]])
                plt.draw(), plt.pause(0.1), plt.clf(), plt.show()

            plt.ion()

            for i in range(len(history)):
                x = get_position(history[i], 0)
                y = get_position(history[i], 1)

                show_current_plot('r')
                show_current_plot('b')
                show_current_plot('r')

        def ffa_function(p):
            x, y = p[0], p[1]
            expr = math.exp(-(x - 4) ** 2 - (y - 4) ** 2) + math.exp(-(x + 4) ** 2 - (y - 4) ** 2)
            expr += 2 * math.exp(-x ** 2 - (y + 4) ** 2) + 2 * math.exp(-x ** 2 - y ** 2)
            return expr

        ffa = FireflyAlgorithm(
            name='ffa', bounds=[-5, 5], n_fireflies=40,
            verbose=True, max_gen=50, f_function=ffa_function,
            mopt='m', distance=manhattan_distance,
        )

        _, _, history1 = ffa.solve()
        draw_plot(ffa.bounds, history1)
