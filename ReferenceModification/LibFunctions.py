import numpy as np
from matplotlib import  pyplot as plt
import math
import cmath
import yaml
from argparse import Namespace

def load_conf(path, fname):
    full_path = path + '/config/' + fname + '.yaml'
    with open(full_path) as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    conf = Namespace(**conf_dict)

    return conf



def add_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] + x2[i] * dx
    return np.array(ret)


def get_distance(x1=[0, 0], x2=[0, 0]):
    d = [0.0, 0.0]
    for i in range(2):
        d[i] = x1[i] - x2[i]
    return np.linalg.norm(d)
     
def sub_locations(x1=[0, 0], x2=[0, 0], dx=1):
    # dx is a scaling factor
    ret = [0.0, 0.0]
    for i in range(2):
        ret[i] = x1[i] - x2[i] * dx
    return ret

def get_gradient(x1=[0, 0], x2=[0, 0]):
    t = (x1[1] - x2[1])
    b = (x1[0] - x2[0])
    if b != 0:
        return t / b
    return 1000000 # near infinite gradient. 

def transform_coords(x=[0, 0], theta=np.pi):
    # i want this function to transform coords from one coord system to another
    new_x = x[0] * np.cos(theta) - x[1] * np.sin(theta)
    new_y = x[0] * np.sin(theta) + x[1] * np.cos(theta)

    return np.array([new_x, new_y])

def normalise_coords(x=[0, 0]):
    r = x[0]/x[1]
    y = np.sqrt(1/(1+r**2)) * abs(x[1]) / x[1] # carries the sign
    x = y * r
    return [x, y]

def get_bearing(x1=[0, 0], x2=[0, 0]):
    grad = get_gradient(x1, x2)
    dx = x2[0] - x1[0]
    th_start_end = np.arctan(grad)
    if dx == 0:
        if x2[1] - x1[1] > 0:
            th_start_end = 0
        else:
            th_start_end = np.pi
    elif th_start_end > 0:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = -np.pi/2 - th_start_end
    else:
        if dx > 0:
            th_start_end = np.pi / 2 - th_start_end
        else:
            th_start_end = - np.pi/2 - th_start_end

    return th_start_end

"""Consider moving to this for more robust solution"""
# def get_bearing1(x0, x1):
#     dy = x1[1] - x0[1]
#     dx = x1[0] - x0[0]

#     bearing = np.arctan2(dy, dx)

#     return bearing

def find_sign(x):
    if x == 0:
        return 1
    return abs(x) / x

def theta_to_xy(theta):
    x = np.sin(theta)
    y = np.cos(theta)

    return np.array([x, y])

def get_rands(a=100, b=0):
    r = [np.random.random() * a + b, np.random.random() * a + b]
    return r

def get_rand_ints(a=100, b=0):
    r = [int(np.random.random() * a + b), int(np.random.random() * a + b)]
    return np.array(r)

def get_rand_coords(xa, xb, ya, yb):
    r = [np.random.random() * xa + xb, np.random.random() * ya + yb]
    return np.array(r)
    

def limit_theta(theta):
    if theta > np.pi:
        theta = theta - 2*np.pi
    elif theta < -np.pi:
        theta += 2*np.pi

    return theta

def add_angles_complex(a1, a2):
    real = math.cos(a1) * math.cos(a2) - math.sin(a1) * math.sin(a2)
    im = math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase
    
def sub_angles_complex(a1, a2): 
    real = math.cos(a1) * math.cos(a2) + math.sin(a1) * math.sin(a2)
    im = - math.cos(a1) * math.sin(a2) + math.sin(a1) * math.cos(a2)

    cpx = complex(real, im)
    phase = cmath.phase(cpx)

    return phase
    
# def sub_angles_complex(a1, a2):
#     c1 = complex(math.cos(a1), math.sin(a1))
#     c2 = complex(math.cos(a2), math.sin(a2))

#     sum_c = c1 * c2.conjugate()
#     phase = cmath.phase(sum_c)

#     return phase

def limit_multi_theta(thetas):
    ths = []
    for theta in thetas:
        th = limit_theta(theta)
        ths.append(th)
    ret_th = np.array(ths)
    return ret_th

def plot(values, moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    moving_avg = moving_average(values, moving_avg_period)
    plt.plot(moving_avg)    
    moving_avg = moving_average(values, moving_avg_period * 5)
    plt.plot(moving_avg)    
    plt.pause(0.001)

def plot_no_avg(values, moving_avg_period=10, title="Results", figure_n=2):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(values)

    plt.pause(0.0001)

def moving_average(data, period):
    return np.convolve(data, np.ones(period), 'same') / period


def plot_multi(value_array, title="Results", figure_n=2, ylim=[-1, 1]):
    plt.figure(figure_n)
    plt.clf()        
    plt.title(title)
    plt.xlabel('Episode')
    plt.ylabel('Duration')

    value_array = np.array(value_array)

    plt.ylim(ylim)

    n_sets = len(value_array[0])
    leg = []
    for i in range(n_sets):
        plt.plot(value_array[:, i])
        leg.append(f"{i}")
    
    plt.legend(leg)

    # plt.plot(values)
    plt.pause(0.0001)

def plot_race_line(track, nset=None, wait=False):
    c_line = track[:, 0:2]
    l_line = c_line - np.array([track[:, 2] * track[:, 4], track[:, 3] * track[:, 4]]).T
    r_line = c_line + np.array([track[:, 2] * track[:, 5], track[:, 3] * track[:, 5]]).T

    plt.figure(1)
    plt.plot(c_line[:, 0], c_line[:, 1], linewidth=2)
    plt.plot(l_line[:, 0], l_line[:, 1], linewidth=1)
    plt.plot(r_line[:, 0], r_line[:, 1], linewidth=1)

    if nset is not None:
        deviation = np.array([track[:, 2] * nset[:, 0], track[:, 3] * nset[:, 0]]).T
        r_line = track[:, 0:2] + deviation
        plt.plot(r_line[:, 0], r_line[:, 1], linewidth=3)

    plt.pause(0.0001)
    if wait:
        plt.show()


def load_config(fname):
    with open('config/' + fname + '.yaml') as file:
        conf_dict = yaml.load(file, Loader=yaml.FullLoader)

    return conf_dict


"""Testing"""
def test():
    print(add_angles_complex(3, -2.5))
    print(sub_angles_complex(3, -2.5))


if __name__ == "__main__":
    test()