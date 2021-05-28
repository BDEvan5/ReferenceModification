from io import FileIO
import numpy as np 
from scipy import ndimage
from matplotlib import pyplot as plt
import yaml
import csv
from PIL import Image

import ReferenceModification.LibFunctions as lib

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

    def xy_to_row_column(self, pt_xy):
        c = int((pt_xy[0] - self.origin[0]) / self.resolution)
        r = int((pt_xy[1] - self.origin[1]) / self.resolution)

        return c, r

    def check_scan_location(self, pt):
        c, r = self.xy_to_row_column(pt)
        if abs(c) > self.map_width -2 or abs(r) > self.map_height -2:
            return True
        val = self.dt_img[r, c]

        if val < 0.1:
            return True
        return False
    
    def check_plan_location(self, pt):
        c, r = self.xy_to_row_column(pt)
        if abs(c) > self.map_width -2 or abs(r) > self.map_height -2:
            return True
        val = self.dt_img[r, c]

        if val < 0.2:
            return True
        return False

    def convert_positions(self, pts):
        xs, ys = [], []
        for pt in pts:
            x, y = self.xy_to_row_column(pt)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)

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

        self.origin = [0, 0, 0] # for ScanSimulator
        
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
            raise FileIO("Problem loading map yaml file")

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

    def xy_to_row_column(self, pt):
        c = int(round(np.clip(pt[0] / self.resolution, 0, self.map_width-2)))
        r = int(round(np.clip(pt[1] / self.resolution, 0, self.map_height-2)))
        return c, r

    def check_scan_location(self, x_in):
        if x_in[0] < 0 or x_in[1] < 0:
            return True
        if x_in[0] > self.forest_width or x_in[1] > self.forest_length:
            return True
        x, y = self.xy_to_row_column(x_in)
        if self.map_img[x, y]:
            return True

    def check_plan_location(self, x_in):
        if x_in[0] < 0 or x_in[1] < 0:
            return True
        if x_in[0] > self.forest_width or x_in[1] > self.forest_length:
            return True
        x, y = self.xy_to_row_column(x_in)
        #TODO: figure out the x, y relationship
        # if self.dt_img[x, y] < 0.2:
        if self.dt_img[y, x] < 0.2:
            return True

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
