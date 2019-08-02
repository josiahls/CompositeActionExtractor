import numpy as np
from math import factorial


def p(element, element_set: np.array):
    return np.sum(element_set == element) / len(element_set)


# noinspection PyUnresolvedReferences


def entropy_regular(x_m, x=None):
    """
    Basic Entropy Analysis using Frequentest Approach.
    >>> R = - sum(p(x_i, x) * np.log(p(x_i, x)) for x_i in x_m)

    """
    x_m = x_m if type(x_m) is np.array else np.array(x_m)
    if x is not None:
        return - sum(p(x_i, x) * np.log(p(x_i, x)) for x_i in x_m if x_i > 0)
    else:
        # If x is None, then we will assume that x_m is already a set of probabilities
        return - sum(x_i * np.log(x_i) for x_i in x_m if x_i > 0)


def entropy_shannon(time_series):
    """Return the Shannon Entropy of the sample data.
    Args:
        time_series: Vector or string of the sample data
    Returns:
        The Shannon Entropy as float value
    """

    # Check if string
    if not isinstance(time_series, str):
        time_series = list(time_series)

    # Create a frequency data
    data_set = list(set(time_series))
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in time_series:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(time_series))

    # Shannon entropy
    # ent = 0.0
    # for freq in freq_list:
    #     ent += freq * np.log2(freq)
    # ent = -ent
    ent = -1 * sum([freq * np.log2(freq) for freq in freq_list if freq > 0])
    return ent


def entropy_approximate(U, m, r):
    """https://en.wikipedia.org/wiki/Approximate_entropy"""
    def _maxdist(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if _maxdist(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (N - m + 1.0) ** (-1) * sum(np.log(C))
    N = len(U)
    return abs(_phi(m + 1) - _phi(m))


def entropy_sample(U, m, r):
    """https://en.wikipedia.org/wiki/Sample_entropy"""
    def _maxdist(x_i, x_j):
        result = max([abs(ua - va) for ua, va in zip(x_i, x_j)])
        return result

    def _phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = 1. * np.array(
            [len([1 for j in range(len(x)) if i != j and _maxdist(x[i], x[j]) <= r]) for i in range(len(x))])
        return sum(C)

    N = len(U)

    return -np.log(_phi(m + 1) / _phi(m))


# From: https://github.com/nikdon/pyEntropy/blob/master/pyentrp/entropy.py
def entropy_permutation(time_series, order=3, delay=1, normalize=False):
    """Permutation Entropy.
    Parameters
    ----------
    time_series : list or np.array
        Time series
    order : int
        Order of permutation entropy
    delay : int
        Time delay
    normalize : bool
        If True, divide by log2(factorial(m)) to normalize the entropy
        between 0 and 1. Otherwise, return the permutation entropy in bit.
    Returns
    -------
    pe : float
        Permutation Entropy
    References
    ----------
    .. [1] Massimiliano Zanin et al. Permutation Entropy and Its Main
        Biomedical and Econophysics Applications: A Review.
        http://www.mdpi.com/1099-4300/14/8/1553/pdf
    .. [2] Christoph Bandt and Bernd Pompe. Permutation entropy — a natural
        complexity measure for time series.
        http://stubber.math-inf.uni-greifswald.de/pub/full/prep/2001/11.pdf
    Notes
    -----
    Last updated (Oct 2018) by Raphael Vallat (raphaelvallat9@gmail.com):
    - Major speed improvements
    - Use of base 2 instead of base e
    - Added normalization
    Examples
    --------
    1. Permutation entropy with order 2
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value between 0 and log2(factorial(order))
        >>> print(entropy_permutation(x, order=2))
            0.918
    2. Normalized permutation entropy with order 3
        >>> x = [4, 7, 9, 10, 6, 11, 3]
        >>> # Return a value comprised between 0 and 1.
        >>> print(entropy_permutation(x, order=3, normalize=True))
            0.589
    """

    def _embed(x, order=3, delay=1):
        """Time-delay embedding.
        Parameters
        ----------
        x : 1d-array, shape (n_times)
            Time series
        order : int
            Embedding dimension (order)
        delay : int
            Delay.
        Returns
        -------
        embedded : ndarray, shape (n_times - (order - 1) * delay, order)
            Embedded time-series.
        """
        N = len(x)
        Y = np.empty((order, N - (order - 1) * delay))
        for i in range(order):
            Y[i] = x[i * delay:i * delay + Y.shape[1]]
        return Y.T

    x = np.array(time_series)
    hashmult = np.power(order, np.arange(order))
    # Embed x and sort the order of permutations
    sorted_idx = _embed(x, order=order, delay=delay).argsort(kind='quicksort')
    # Associate unique integer to each permutations
    hashval = (np.multiply(sorted_idx, hashmult)).sum(1)
    # Return the counts
    _, c = np.unique(hashval, return_counts=True)
    # Use np.true_divide for Python 2 compatibility
    p = np.true_divide(c, c.sum())
    pe = -np.multiply(p, np.log2(p)).sum()
    if normalize:
        pe /= np.log2(factorial(order))
    return pe



