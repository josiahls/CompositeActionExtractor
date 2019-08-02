import matplotlib.pyplot as plt


def entropy_func_comparison(entropy_series, actual_values, title):
    plt.figure(figsize=(15, 5))
    plt.subplot(121)
    plt.title(title)
    plt.plot(entropy_series)
    plt.subplot(122)
    plt.plot(actual_values)
