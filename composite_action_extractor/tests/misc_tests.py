import numpy as np

from utils.misc import *


def test_bin_array():
    some_array = np.array([1, 2, 3, 4.5, .6, 7., 4. , 5.6])

    print(bin_numpy(some_array))

def test_windowed():
    some_array = np.array([1, 2, 3, 4.5, .6, 7., 4. , 5.6])

    print(windowed_numpy(bin_numpy(some_array)))

def test_windowed_distance():
    w = 5

    some_array = np.array([1, 2, 3, 4.5, .6, 7., 4. , 5.6, 4, 5, 7, 3, 1, 5, 6])
    windowed = windowed_numpy(bin_numpy(some_array), w)
    """
    Given an array [1,2,3,4,5,6,7,8,9,10], and w=3, then the first sample:
    
    windows[0] = array[0:3 // 2] => 1, 2
    
    Then the comparison windows are:
    
    """
    print([[windowed[i + 1:i + w]] for i in range(len(windowed) - w // 2)])