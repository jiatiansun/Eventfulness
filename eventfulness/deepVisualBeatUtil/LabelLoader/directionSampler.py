import numpy as np
from matplotlib import pyplot as plt

class DirectionSampler():
    def __init__(self, num_dirs):
        self.num_dirs = num_dirs

    def generate_2D_directions(self):
        theta_step = 2 * np.pi / self.num_dirs
        thetas = np.arange(self.num_dirs) * theta_step
        xs = np.cos(thetas)
        ys = np.sin(thetas)
        return np.stack([xs, ys])




