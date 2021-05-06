import numpy as np
from numba import njit

@njit(cache=True)
def xy_to_row_column(pt_xy, origin, resolution):
    c = (pt_xy[0] - origin[0]) / resolution
    c = int((pt_xy[0] - origin[0]) / resolution)
    r = int((pt_xy[1] - origin[1]) / resolution)

    return c, r

@njit(cache=True)
def check_scan_location(pt, origin, resolution, map_width, map_height, dt_img, val=0.1):
    c, r = xy_to_row_column(pt, origin, resolution)
    if abs(c) > map_width -2 or abs(r) > map_height -2:
        return True

    if dt_img[r, c] < val:
        return True
    
    return False

@njit(cache=True)
def convert_positions(pts, origin, resolution):
    n = len(pts)
    xs, ys = np.zeros(n), np.zeros(n)
    for i, pt in enumerate(pts):
        xs[i], ys[i] = xy_to_row_column(pt, origin, resolution)

    return xs, ys



