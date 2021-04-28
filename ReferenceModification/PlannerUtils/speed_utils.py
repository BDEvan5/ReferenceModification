import numpy as np
from numba import njit



@njit(cache=True)
def calculate_speed(delta):
    b = 0.523
    g = 9.81
    l_d = 0.329
    f_s = 0.5
    max_v = 6

    if abs(delta) < 0.06:
        return max_v

    V = f_s * np.sqrt(b*g*l_d/np.tan(abs(delta)))

    return V