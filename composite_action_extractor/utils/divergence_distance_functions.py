import numpy as np
from scipy.stats import entropy, wasserstein_distance, kstest


def kl_divergence(p: np.array, q: np.array):
    return entropy(p, q)


def jensen_shannon_divergence(p: np.array, q: np.array):
    """https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence"""
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m) + entropy(q, m))


def total_variation_distance(p: np.array, q: np.array):
  return sum(abs(p-q))/2


def earth_mover_wasserstein_distance(p, q):
    return wasserstein_distance(p, q)


def ks_test_distance(f_1, f_2):
    """
    Taken from https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test

    Notes:
        Not added yet. I want to compare this to other normality tests.

    Args:
        f_1:
        f_2:

    Returns:

    """
    return kstest(f_1, f_2)[0]
