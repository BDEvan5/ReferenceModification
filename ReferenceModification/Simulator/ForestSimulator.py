
import csv 
from scipy import ndimage
from matplotlib import pyplot as plt
import yaml
from PIL import Image 
import numpy as np
from numba import njit
import os

import ReferenceModification.LibFunctions as lib
from ReferenceModification.Simulator.map_utils import *

from ReferenceModification.Simulator.BaseSimulatorClasses import BaseSim



class ForestMap:
    def __init__(self, map_name) -> None:
        self.map_name = map_name 

        # map info
        self.resolution = None
        self.n_obs = None 
        self.map_height = None
        self.map_width = None
        self.forest_length = None
        self.forest_width = None
        self.start_pose = None
        self.obs_size = None
        self.obstacle_buffer = None
        self.end_y = None
        self.end_goal = None

        self.origin = np.zeros(3) # for ScanSimulator
        
        self.dt_img = None
        self.map_img = None

        self.ref_pts = None # std wpts that aren't expanded
        self.ss_normal = None # not expanded
        self.diffs = None
        self.l2s = None

        self.load_map()
        self.load_center_pts()

    def load_map(self):
        file_name = 'maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())

        try:
            self.resolution = yaml_file['resolution']
            self.n_obs = yaml_file['n_obs']
            self.obs_size = yaml_file['obs_size']
            self.start_pose = np.array(yaml_file['start_pose'])
            self.forest_length = yaml_file['forest_length']
            self.forest_width = yaml_file['forest_width']
            self.obstacle_buffer = yaml_file['obstacle_buffer']
            self.end_y = yaml_file['end_y']
        except Exception as e:
            print(e)
            raise IOError("Problem loading map yaml file")

        self.end_goal = np.array([self.start_pose[0], self.end_y])

        self.map_height = int(self.forest_length / self.resolution)
        self.map_width = int(self.forest_width / self.resolution)
        self.map_img = np.zeros((self.map_width, self.map_height))

        self.set_dt()

    def add_obstacles(self):
        self.map_img = np.zeros((self.map_width, self.map_height))

        y_length = (self.end_y - self.obstacle_buffer*2 - self.start_pose[1] - self.obs_size)
        box_factor = 1.4
        y_box = y_length / (self.n_obs * box_factor)
        rands = np.random.random((self.n_obs, 2))
        xs = rands[:, 0] * (self.forest_width-self.obs_size) 
        ys = rands[:, 1] * y_box
        y_start = self.start_pose[1] + self.obstacle_buffer
        y_pts = [y_start + y_box * box_factor * i for i in range(self.n_obs)]
        ys = ys + y_pts

        obs_locations = np.concatenate([xs[:, None], ys[:, None]], axis=-1)
        obs_size_px = int(self.obs_size/self.resolution)
        for location in obs_locations:
            x, y = self.xy_to_row_column(location)
            # print(f"Obstacle: ({location}): {x}, {y}")
            self.map_img[x:x+obs_size_px, y:y+obs_size_px] = 1
        
    def set_dt(self):
        img = np.ones_like(self.map_img) - self.map_img
        img[0, :] = 0 #TODO: move this to the original map img that I make
        img[-1, :] = 0
        img[:, 0] = 0
        img[:, -1] = 0

        self.dt_img = ndimage.distance_transform_edt(img) * self.resolution
        self.dt_img = np.array(self.dt_img).T

        return self.dt_img

    def check_plan_location(self, pt):
        return check_scan_location(pt, self.origin, self.resolution, self.map_width, self.map_height, self.dt_img, 0.2)

    def check_scan_location(self, pt):
        return check_scan_location(pt, self.origin, self.resolution, self.map_width, self.map_height, self.dt_img, 0.1)

    def convert_positions(self, pts):
        return convert_positions(pts, self.origin, self.resolution)

    def xy_to_row_column(self, pt):
        return xy_to_row_column(pt, self.origin, self.resolution)

    def render_map(self, figure_n=1, wait=False):
        #TODO: draw the track boundaries nicely
        f = plt.figure(figure_n)
        plt.clf()

        plt.xlim([0, self.map_width])
        plt.ylim([0, self.map_height])

        plt.imshow(self.map_img.T, origin='lower')

        xs = np.linspace(0, self.map_width, 10)
        ys = np.ones_like(xs) * self.end_y / self.resolution
        plt.plot(xs, ys, '--')     
        x, y = self.xy_to_row_column(self.start_pose[0:2])
        plt.plot(x, y, '*', markersize=14)

        plt.pause(0.0001)
        if wait:
            plt.show()
            pass

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

    def render_wpts(self, wpts):
        plt.figure(4)
        xs, ys = self.convert_positions(wpts)
        plt.plot(xs, ys, '--', linewidth=2)
        # plt.plot(xs, ys, '+', markersize=12)

        plt.pause(0.0001)

    def render_aim_pts(self, pts):
        plt.figure(4)
        xs, ys = self.convert_positions(pts)
        # plt.plot(xs, ys, '--', linewidth=2)
        plt.plot(xs, ys, 'x', markersize=10)

        plt.pause(0.0001)

    def load_center_pts(self):

        track = []
        filename = 'maps/' + self.map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        self.ref_pts = track[:, 1:3]
        self.ss_normal = track[:, 0]
        # self.expand_wpts()
        # print(self.ref_pts)
        self.diffs = self.ref_pts[1:,:] - self.ref_pts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2

    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.wpts
        new_line = []
        for i in range(len(self.wpts)-1):
            dd = lib.sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = lib.add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

        self.wpts = np.array(new_line)



class ForestSim(BaseSim):
    """
    Simulator for Race Tracks

    Data members:
        map_name: name of the map to be used. Forest yaml file which stores the parameters for the forest. No image is required.

    """
    def __init__(self, map_name, sim_conf=None):
        """
        Init function

        Args:
            map_name: name of forest map to use.
            sim_conf: config file for simulation
        """
        if sim_conf is None:
            # path = os.path.dirname(__file__)
            # sim_conf = lib.load_conf(path, "std_config")
            sim_conf = lib.load_conf("std_config")

        env_map = ForestMap(map_name)
        BaseSim.__init__(self, env_map, self.check_done_forest, sim_conf
        )

    def check_done_forest(self):
        """
        Checks if the episode in the forest is complete 

        Returns:
            done (bool): a flag if the ep is done
        """
        self.reward = 0 # normal
        # check if finished lap
        dx = self.state[0] - self.env_map.start_pose[0]
        dx_lim = self.env_map.forest_width * 0.5
        if dx < dx_lim and self.state[1] > self.env_map.end_y:
            self.done = True
            self.reward = 1
            self.done_reason = f"Lap complete"

        # check crash
        elif self.env_map.check_scan_location(self.state[0:2]):
            self.done = True
            self.reward = -1
            self.done_reason = f"Crash obstacle: [{self.state[0]:.2f}, {self.state[1]:.2f}]"

        # check steps
        elif self.steps > self.max_steps:
            self.done = True
            self.reward = -1
            self.done_reason = f"Max steps"
        # check orientation
        elif abs(self.state[2]) > 0.66*np.pi:
            self.done = True
            self.done_reason = f"Vehicle turned around"
            self.reward = -1

        return self.done




