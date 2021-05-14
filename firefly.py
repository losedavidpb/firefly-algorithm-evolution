# ---------------------------------------------------------------
# ---- All Copyright reserved (c) by David Parreño Barbuzano ----
# ---------------------------------------------------------------

from numpy.random import randint
import numpy as np

from utils import euclidean_distance, michalewicz

class Firefly(object):

    def __init__(self, **kwargs):
        """Constructs a new instance of `Firefly` class.

        Valid parameters:
            * dim               -- Dimension of the problem (default: 2)
            * bounds            -- Bounds of valid solutions (default: [0, 4])
            * position          -- Position of the firefly (default: random position of "dim" size)
            * alpha             -- Randomness coefficient (default: 0.2)
            * beta_0            -- Initial attractiveness coefficient (default: 1)
            * beta_min          -- Min attractiveness coefficient (default: 0.2)
            * gamma             -- Absorption light coefficient (default: 1)
            * distance          -- Distance function (default: euclidean_distance)
            * f_function        -- Fitness function (default: michalewicz)
            * light             -- Light intensity of the firefly (default: michalewicz(self.position))
            * mopt              -- Minimization method selected (default: m)

        """
        self.dim = kwargs.get('dim', 2)
        self.mopt = kwargs.get('dim', 'm')

        self.bounds = np.asarray(kwargs.get('bounds', [0, 4]))
        self.position = kwargs.get('position', None)

        if self.position is None:
            self.position = np.zeros(shape=(1, self.dim)).flatten()

            for i in range(self.position.size):
                low, high = self.bounds[0], self.bounds[1]
                self.position[i] = np.random.uniform(low, high)

        self.distance = kwargs.get('distance', euclidean_distance)
        self.f_function = kwargs.get('f_function', michalewicz)
        self.light = self.f_function(self.position)

        self.alpha = kwargs.get('alpha', 0.2)
        self.gamma = kwargs.get('gamma', 1)
        self.beta_0 = kwargs.get('beta_0', 1)
        self.beta_min = kwargs.get('beta_min', 0.2)

    def __repr__(self):
        str_format = 'Firefly{}: light={}'
        return str.format(str_format, self.position, self.light)

    def __str__(self):
        str_format = 'Firefly{}: light={}'
        return str.format(str_format, self.position, self.light)

    def attractiveness(self, r):
        """Returns current firefly's attractiveness. """
        return (self.beta_0 - self.beta_min) * np.exp(-self.gamma * r * r) + self.beta_min
        # return self.beta_0 * np.exp(-self.gamma * r * r)
        # return self.beta_0 * (1 + self.gamma * r * r) # Better solution!!

    def randomness(self):
        """Returns current randomness value. """
        # scale = np.abs(self.bounds[1] - self.bounds[0])
        # return self.alpha * (np.random.uniform(1, self.dim) - 0.5) * scale
        return self.alpha * (np.random.uniform(0, 1) - 0.5)

    def put_on_range(self):
        """Set firefly's position between current bounds. """
        for pos in range(self.position.size):
            if self.position[pos] <= self.bounds[0]:
                self.position[pos] = self.bounds[0]

            if self.position[pos] >= self.bounds[1]:
                self.position[pos] = self.bounds[1]

    def move_towards(self, better):
        """Move firefly to passed one considering distance and randomness. """
        r = self.distance(self.position, better.position)
        beta_temp = self.attractiveness(r)

        # self.position = self.position + beta_temp * (other_firefly.position - self.position) + self.randomness()
        self.position = self.position * (1 - beta_temp)
        self.position += better.position * beta_temp + self.randomness()

        self.put_on_range()
        self.light = self.f_function(self.position)

    def move_random(self, dist):
        """ Move current firefly randomly. """
        self.position = np.random.uniform(self.position - dist, self.position + dist, self.position.size)

        self.put_on_range()
        self.light = self.f_function(self.position)
