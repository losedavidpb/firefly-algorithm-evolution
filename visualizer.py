# ---------------------------------------------------------------
# ---- All Copyright reserved (c) by David Parreño Barbuzano ----
# ---------------------------------------------------------------

from abc import ABC, abstractmethod
from matplotlib import pyplot as plt, animation
import numpy as np

class VisualizerFFA(ABC):

    def __init__(self, solver, file_path=None, interval_time=2000):
        """Constructs a new instance of `VisualizerFFA` class.

        Valid parameters:
            * solver            -- Firefly algorithm solver that will be shown
            * file_path         -- Valid path on which the animation will be stored
            * interval_time     -- Interval time in ms for each frame

        """
        self.solver = solver
        self.interval_time = interval_time
        self.file_path = file_path
        self.history = []

        self.lower_bound = np.round(self.solver.bounds[0]).astype('int')
        self.upper_bound = np.round(self.solver.bounds[1]).astype('int')

        x = np.linspace(self.solver.bounds[0], self.solver.bounds[1], 400)
        y = np.linspace(self.solver.bounds[0], self.solver.bounds[1], 400)
        self.X, self.Y = np.meshgrid(x, y)
        self.Z = np.apply_along_axis(self.solver.f_function, 0, [self.X, self.Y])

        self.fig, self.ax = plt.subplots()
        self.pff, = self.ax.plot([], [], '.')

    @abstractmethod
    def init(self):
        """Initializes visualizer with static or default settings. """
        pass

    @abstractmethod
    def update(self, frame):
        """Update passed frame with the fireflies of current generation. """
        pass

    @abstractmethod
    def show(self):
        """Start the animation of the visualizer. """
        pass

class VisualizerFFA2D(VisualizerFFA):

    def init(self):
        self.ax.set_xlim(self.lower_bound, self.upper_bound)
        self.ax.set_ylim(self.lower_bound, self.upper_bound)

        if self.upper_bound <= 100:
            plt.xticks([i for i in range(self.lower_bound, self.upper_bound + 1)])
            plt.yticks([i for i in range(self.lower_bound, self.upper_bound + 1)])

        self.ax.set_title('Evolution of Firefly Algorithm', loc='left')
        self.ax.set_title('Generation 0', loc='right')
        plt.grid()

        plt.contourf(self.X, self.Y, self.Z)
        cp = plt.contour(self.X, self.Y, self.Z, origin='lower', extend='both', linewidths=2)
        plt.colorbar(cp, shrink=0.8, orientation='vertical')

        best_firefly, self.history = self.solver.solve()

        if self.solver.verbose:
            print(str.format('BEST FIREFLY: {}', best_firefly))

        return self.pff,

    def update(self, frame):
        color = 'r' if frame % 2 == 0 else 'b'
        swarm = self.history[0]
        self.history.remove(self.history[0])

        x = [f.position[0] for f in swarm]
        y = [f.position[1] for f in swarm]

        self.pff.set_color(color)
        self.pff.set_data(x, y)

        self.ax.set_title("Generation {}".format(frame + 1), loc='right')
        return self.pff,

    def show(self):
        vanim = animation.FuncAnimation(
            fig=self.fig, frames=self.solver.max_gen, interval=self.interval_time, blit=False,
            func=self.update, init_func=self.init, repeat=False)

        plt.show(block=True)

        if self.file_path is not None:
            self.solver.verbose = False
            vanim.save(self.file_path, writer='pillow', fps=3, dpi=100)
