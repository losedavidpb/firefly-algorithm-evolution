# ---------------------------------------------------------------
# ---- All Copyright reserved (c) by David Parreño Barbuzano ----
# ---------------------------------------------------------------

from utils import euclidean_distance
from firefly import Firefly
from copy import deepcopy

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

        opt_functions = {'m': lambda oc, nc: nc < oc, 'M': lambda oc, nc: nc > oc}
        self.optimize = opt_functions[self.mopt]

    def init_swarm(self):
        """Initialize random swarm population of fireflies. """
        swarm_population = []

        for i in range(self.n_fireflies):
            firefly = Firefly(
                dim=self.dim, bounds=self.bounds, alpha=self.alpha,
                gamma=self.gamma, beta_min=self.beta_min, distance=self.distance,
                beta_0=self.beta_0, f_function=self.f_function
            )

            swarm_population.append(firefly)

        return swarm_population

    def reduce_alpha(self, swarm):
        """Reduce randomness coefficient. """
        # self.delta = 1 - (10 ** (-4) / 0.9) ** (1 / self.max_gen)
        # self.alpha = (1 - self.delta) * self.alpha
        self.alpha = self.alpha * self.delta

        for i in range(len(swarm)):
            swarm[i].alpha = self.alpha

    def solve(self):
        """Implements firefly algorithm for current problem.

        Since intermediate solutions could be used for animations,
        this function would store swarm's history, so returned
        values are best cost, best firefly, and history.

        """
        reverse_val = self.mopt == 'M'

        # Initial swarm of random fireflies
        swarm = self.init_swarm()
        swarm.sort(key=lambda f: f.light, reverse=reverse_val)

        # Firefly with the best solution and historical movements
        best_firefly, history_fireflies = deepcopy(swarm[0]), []

        if self.verbose is True:
            print(str.format(">> Evolution for FFA {} <<", self.name))

        for gen in range(self.max_gen):
            if self.verbose is True:
                str_format = 'Generation {}: best_firefly={}'
                print(str.format(str_format, gen + 1, best_firefly))
                for firefly in swarm: print('\t *', firefly)

            for i in range(0, self.n_fireflies):
                for j in range(0, self.n_fireflies):
                    # Light would be more attractive
                    # depending on the optimization method
                    if self.optimize(swarm[i].light, swarm[j].light):
                        swarm[i].move_towards(swarm[j])

            swarm.sort(key=lambda f: f.light, reverse=reverse_val)
            for k in range(len(swarm)): swarm[k].move_random(0.1)

            if self.optimize(swarm[0].light, best_firefly.light):
                best_firefly = deepcopy(swarm[0])

            history_fireflies.append(deepcopy(swarm))
            self.reduce_alpha(swarm)

        return best_firefly, history_fireflies
