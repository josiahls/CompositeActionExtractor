from utils.divergence_distance_functions import jensen_shannon_divergence
from utils.entropy_functions import entropy_shannon
import numpy as np


def test_entropy_regular():

    x = [1, 1, 1, 2, 2, 2, 3, 4, 3, 5, 3]
    print(entropy_shannon(x))


def test_div_regular():

    x = np.array([0.1, 0.2, 0.3, 0.1, 0.2, 0.3]).astype(float)
    print(jensen_shannon_divergence(x[0:3], x[1:4]))
