import numpy as np

def euclidean_distance(pos_1, pos_2):
    """Returns euclidean distance between two positions. """
    result = [(x1 - x2)**2 for x1, x2 in zip(pos_1, pos_2)]
    return np.math.sqrt(sum(result))

def manhattan_distance(pos_1, pos_2):
    """Returns manhattan distance between two positions. """
    result = [(x1 - x2) for x1, x2 in zip(pos_1, pos_2)]
    return sum(result)
