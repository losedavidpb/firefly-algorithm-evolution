# ---------------------------------------------------------------
# ---- All Copyright reserved (c) by David Parreño Barbuzano ----
# ---------------------------------------------------------------

from numpy.random import randint
import numpy as np

from distance_functions import euclidean_distance

class Firefly(object):

    def __init__(self, **kwargs):
        """Constructs a new instance of `Firefly` class.

        Valid parameters:
            * dim               -- Dimension of the problem (default: 2)
            * bounds            -- Bounds of valid solutions (default: [0, 4])
            * position          -- Position of the firefly (default: random position of "dim" size)
            * fitness_value     -- Fitness value of the problem (default: 0)
            * light             -- Light intensity of the firefly based on distance (default: 1)
            * alpha             -- Randomness coefficient (default: 0.2)
            * beta              -- Attractiveness coefficient (default: 1)
            * beta_0            -- Initial attractiveness coefficient (default: 1)
            * beta_min          -- Min attractiveness coefficient (default: 0.2)
            * gamma             -- Absorption light coefficient (default: 1)
            * distance          -- Distance function (default: euclidean_distance)

        """
        self.dim = kwargs.get('dim', 2)
        self.bounds = kwargs.get('bounds', [0, 4])
        self.position = kwargs.get('position', randint(self.bounds[0], self.bounds[1], (1, self.dim)).astype("float"))
        self.position = (np.asarray(self.position)).flatten()

        self.fitness_value = kwargs.get('fitness_value', 0)
        self.light = kwargs.get('light', 1)
        if self.fitness_value != 0: self.light = self.fitness_value

        self.alpha = kwargs.get('alpha', 0.2)
        self.beta = kwargs.get('beta', 1)
        self.beta_0 = kwargs.get('beta_0', 1)
        self.beta_min = kwargs.get('beta_min', 0.2)
        self.gamma = kwargs.get('gamma', 1)

        self.distance = kwargs.get('distance', euclidean_distance)

    def __repr__(self):
        str_format = 'Firefly{}: fitness_value={}, alpha={}, beta={}, gamma={}'
        return str.format(str_format, self.position, self.fitness_value, self.alpha, self.beta, self.gamma)

    def __str__(self):
        str_format = 'Firefly{}: fitness_value={}, alpha={}, beta={}, gamma={}'
        return str.format(str_format, self.position, self.fitness_value, self.alpha, self.beta, self.gamma)

    def light_intensity(self, r):
        """Returns current firefly's light intensity. """
        light_0 = self.fitness_value
        return light_0 * np.exp(-self.gamma * r * r)
        # return light_0 / (1 + self.gamma * r * r) # Better solution!!

    def attractiveness(self, r):
        """Returns current firefly's attractiveness. """
        return (self.beta_0 - self.beta_min) * np.exp(-self.gamma * r * r) + self.beta_min
        # return self.beta * np.exp(-self.gamma * r * r)
        # return self.beta * (1 + self.gamma * r * r) # Better solution!!

    def randomness(self):
        """Returns current randomness value. """
        scale = np.abs(self.bounds[1] - self.bounds[0])
        return self.alpha * (np.random.uniform(1, self.dim) - 0.5) * scale
        # return self.alpha * (np.random.uniform(0, 1) - 0.5)

    def put_on_range(self, pos):
        """Set firefly's position between current bounds using randomness. """
        if self.position[pos] < self.bounds[0]:
            self.position[pos] = randint(self.bounds[0], self.bounds[1])
        if self.position[pos] > self.bounds[1]:
            self.position[pos] = randint(self.bounds[0], self.bounds[1])

    def move_towards(self, other_firefly):
        """Move firefly to passed one considering distance and randomness. """
        r = self.distance(self.position, other_firefly.position)
        beta_temp = self.attractiveness(r)

        for i, xi, xj in zip(range(self.position.size), self.position, other_firefly.position):
            self.position[i] = xi + beta_temp * (xj - xi) + self.randomness()
            # self.position[i] = xi * (1 - beta_temp) + xj * beta_temp + self.randomness() # Better solution!!
            self.put_on_range(i)

        self.light = self.light_intensity(r)

    def random_move(self):
        """Move firefly randomly using randomness coefficient. """
        for i in range(self.position.size):
            self.position[i] = self.position[i] + self.randomness()
            self.put_on_range(i)
