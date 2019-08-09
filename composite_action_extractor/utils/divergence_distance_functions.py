import numpy as np
from scipy.stats import entropy, wasserstein_distance, kstest


def kl_divergence(p: np.array, q: np.array):
    return entropy(p, q)


def jensen_shannon_divergence(p: np.array, q: np.array):
    """Calculates the Jensen Shannon Divergence of a probability distribution.


    Args:
        p (np.array): Probability distribution of window size w
        q (np.array): Comparison distributions of window size w, but offset.
    References:
        [1] Lin, Jianhua. "Divergence measures based on the Shannon entropy."
        IEEE Transactions on Information theory 37.1 (1991): 145-151.
        [2] "Jensen–Shannon Divergence." En.wikipedia.org. N. p., 2019. Web. 9 Aug. 2019.
        https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence

    Returns (float): value representing Jensen Shannon Divergence.
    """
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
    raise NotImplementedError
    # return kstest(f_1, f_2)[0]
