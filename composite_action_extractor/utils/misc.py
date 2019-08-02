import numpy as np
from ast import literal_eval


def df_to_numpy(list_of_dfs: list):
    list_dict_of_numpy = []
    for episode in list_of_dfs:
        local_array = {name: episode[name].values for name in episode}
        local_array = {name: list_of_str_to_num(local_array[name]) for name in local_array}

        list_dict_of_numpy.append(local_array)

    return list_dict_of_numpy


def list_of_str_to_num(list_of_str):
    return np.array([literal_eval(_) if type(_) is str else _ for _ in list_of_str])


def convert_df_col_to_np(list_of_values: list):
    values = None
    for element in list_of_values:
        values = element if values is None else np.vstack((values, element))
    return values


def bin_numpy(array: np.array, bins=64):
    """
    Returns a version of the array with the indexes of the bins that each element belongs to.

    :param array:
    :param bins:
    :return:
    """
    return np.array([np.histogram(element, bins=bins, range=(array.min(), array.max()))[0].argmax()
                     for element in array])


def windowed_numpy(array: np.array, w=5):
    return [array[i - w // 2:i + w // 2] for i in range(w // 2, len(array) - w // 2)]


def get_frequentest_probability(array: np.array):
    return np.array([np.average(x == array) for x in array])


def get_distance_q(array: np.array, i, w):
    return [array[i + shift:i + w + shift] for shift in range(i+1, i + w + 1)]


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
