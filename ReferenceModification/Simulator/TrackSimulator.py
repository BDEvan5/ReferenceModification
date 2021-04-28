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

class TrackMap:
    def __init__(self, map_name) -> None:
        self.map_name = map_name 

        # map info
        self.resolution = None
        self.origin = None
        self.n_obs = None 
        self.map_height = None
        self.map_width = None
        self.start_pose = None
        self.obs_size = None
        self.end_goal = None
        
        self.map_img = None
        self.dt_img = None
        self.obs_img = None #TODO: combine to single image with dt for faster scan

        self.load_map()

        self.ss = None
        self.wpts = None
        self.t_pts = None
        self.nvecs = None
        self.ws = None 
        self.ref_pts = None # std wpts that aren't expanded
        self.ss_normal = None # not expanded
        self.diffs = None
        self.l2s = None

        try:
            # raise FileNotFoundError
            self._load_csv_track()
        except FileNotFoundError:
            print(f"Problem Loading map - generate new one")

    def load_map(self):
        file_name = 'maps/' + self.map_name + '.yaml'
        with open(file_name) as file:
            documents = yaml.full_load(file)
            yaml_file = dict(documents.items())

        try:
            self.resolution = yaml_file['resolution']
            self.origin = yaml_file['origin']
            self.n_obs = yaml_file['n_obs']
            self.obs_size = yaml_file['obs_size']
            map_img_path = 'maps/' + yaml_file['image']
            self.start_pose = np.array(yaml_file['start_pose'])
        except Exception as e:
            print(f"Problem loading, check key: {e}")
            raise FileIO("Problem loading map yaml file")

        self.end_goal = self.start_pose[0:2]

        self.map_img = np.array(Image.open(map_img_path).transpose(Image.FLIP_TOP_BOTTOM))
        self.map_img = self.map_img.astype(np.float64)
        if len(self.map_img.shape) == 3:
            self.map_img = self.map_img[:, :, 0]
        self.obs_img = np.zeros_like(self.map_img) # init's obs img

        # grayscale -> binary
        self.map_img[self.map_img <= 128.] = 0.
        self.map_img[self.map_img > 128.] = 255.

        self.map_height = self.map_img.shape[0]
        self.map_width = self.map_img.shape[1]

        dt = ndimage.distance_transform_edt(self.map_img) 
        self.dt_img = np.array(dt *self.resolution)
    
    def add_obstacles(self):
        obs_img = np.zeros_like(self.obs_img) 
        obs_size_m = np.array([self.obs_size, self.obs_size]) 
        obs_size_px = obs_size_m / self.resolution

        rands = np.random.uniform(size=(self.n_obs, 2))
        idx_rands = rands[:, 0] * len(self.ws)
        w_rands = (rands[:, 1] * 2 - np.ones_like(rands[:, 1]))
        w_rands = w_rands * np.mean(self.ws) * 0.3 # x average length, adjusted to be both sides of track
        # magic 0.8 is to keep the obstacles closer to the center of the track

        obs_locations = []
        for i in range(self.n_obs):
            idx = idx_rands[i]
            w = w_rands[i]
            
            int_idx = int(idx) # note that int always rounds down

            # start with using just int_idx
            n = self.nvecs[i]
            offset = np.array([n[0]*w, n[1]*w])
            location = lib.add_locations(self.t_pts[int_idx], offset)
            if lib.get_distance(location, self.start_pose[0:2]) < 1:
                continue
            # location = np.flip(location)
            # location = self.t_pts[int_idx]
            rc_location = self.xy_to_row_column(location)
            location = np.array(location, dtype=int)
            obs_locations.append(rc_location)


        obs_locations = np.array(obs_locations)
        for location in obs_locations:
            x, y = location[0], location[1]
            for i in range(0, int(obs_size_px[0])):
                for j in range(0, int(obs_size_px[1])):
                    if x+i < self.map_width and y+j < self.map_height:
                        # obs_img[x+i, y+j] = 255
                        obs_img[y+j, x+i] = 255

        self.obs_img = obs_img

    def set_dt(self):
        dt = ndimage.distance_transform_edt(self.map_img - self.obs_img) 
        self.dt_img = np.array(dt *self.resolution)

        return self.dt_img

    def render_map(self, figure_n=4, wait=False):
        f = plt.figure(figure_n)
        plt.clf()

        plt.xlim([0, self.map_width])
        # plt.ylim([self.map_height, 0])


        if self.obs_img is None:
            plt.imshow(self.map_img, origin='lower')
        else:
            plt.imshow(self.obs_img + self.map_img, origin='lower')

        plt.gca().set_aspect('equal', 'datalim')

        if self.wpts is not None:
            xs, ys = self.convert_positions(self.wpts)
            plt.plot(xs, ys, '--')

        plt.pause(0.0001)
        if wait:
            plt.show()

    def _load_csv_track(self):
        track = []
        filename = 'maps/' + self.map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        self.wpts = track[:, 1:3]
        self.ss = track[:, 0]
        self.ss_normal = np.copy(self.ss)
        self.ref_pts = np.copy(self.wpts)
        
        self.diffs = self.ref_pts[1:,:] - self.ref_pts[:-1,:]
        self.l2s   = self.diffs[:,0]**2 + self.diffs[:,1]**2

        self.expand_wpts()

        track = []
        filename = 'maps/' + self.map_name + "_std.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename} in env_map")

        self.t_pts = track[:, 0:2]
        self.nvecs = track[:, 2: 4]
        self.ws = track[:, 4:6]

        # plt.plot(self.t_pts[:, 0], self.t_pts[:, 1])
        # plt.pause(0.001)

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

    def render_wpts(self, wpts):
        plt.figure(4)
        xs, ys = self.convert_positions(wpts)
        plt.plot(xs, ys, '--', linewidth=2)
        # plt.plot(xs, ys, '+', markersize=12)

        plt.pause(0.0001)


class TrackSim(BaseSim):
    """
    Simulator for Race Tracks, inherits from the base sim and adds a layer for use with a race track for f110

    Important to note the check_done function which checks if the episode is complete
        
    """
    def __init__(self, map_name, sim_conf=None):
        """
        Init function

        Args:
            map_name: name of map to use.
            sim_conf: config file for simulation

        """
        if sim_conf is None:
            path = os.path.dirname(__file__)
            sim_conf = lib.load_conf(path, "std_config")

        env_map = TrackMap(map_name)
        BaseSim.__init__(self, env_map, self.check_done_reward_track_train, sim_conf)
        self.end_distance = sim_conf.end_distance

    def check_done_reward_track_train(self):
        """
        Checks if the race lap is complete

        Returns
            Done flag
        """
        self.reward = 0 # normal
        if self.env_map.check_scan_location([self.car.x, self.car.y]):
            self.done = True
            self.colission = True
            self.reward = -1
            self.done_reason = f"Crash obstacle: [{self.car.x:.2f}, {self.car.y:.2f}]"

        if self.steps > self.max_steps:
            self.done = True
            self.done_reason = f"Max steps"

        car = [self.car.x, self.car.y]
        cur_end_dis = lib.get_distance(car, self.env_map.start_pose[0:2]) 
        if cur_end_dis < self.end_distance and self.steps > 800:
            self.done = True
            self.reward = 1
            self.done_reason = f"Lap complete, d: {cur_end_dis}"


        return self.done


