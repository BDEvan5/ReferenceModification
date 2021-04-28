import numpy as np 
import casadi as ca 
import csv
from matplotlib import pyplot as plt
from numba import njit
 
import toy_auto_race.Utils.LibFunctions as lib
from toy_auto_race.lidar_viz import LidarViz
from toy_auto_race.speed_utils import calculate_speed



class TrackFGM:    
    BUBBLE_RADIUS = 250
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 100
    MAX_LIDAR_DIST = 2
    REDUCTION = 100
    
    def __init__(self):
        # used when calculating the angles of the LiDAR data
        self.vis = LidarViz(1000)
        self.degrees_per_elem = None
        self.name = "Follow the Race Gap"
        self.n_beams = 1000
    
    def preprocess_lidar(self, ranges):
        self.degrees_per_elem = np.pi / len(ranges)
        proc_ranges = np.array(ranges[self.REDUCTION:-self.REDUCTION])
        proc_ranges = ranges
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE

        return proc_ranges
   
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE), 'same') 
        averaged_max_gap = averaged_max_gap / self.BEST_POINT_CONV_SIZE
        best = averaged_max_gap.argmax()
        idx = best + start_i

        mid_gap = int(start_i *0.5 + end_i*0.5)
        alpha = 0.9
        # idx = int(idx*alpha + (1-alpha)*mid_gap) 

        return idx

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data
        """
        return (range_index - (range_len/2)) * self.degrees_per_elem 

    def process_lidar(self, ranges):
        proc_ranges = self.preprocess_lidar(ranges)
        closest = proc_ranges.argmin()

        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges)-1
        proc_ranges[min_index:max_index] = 0

        gap_start, gap_end = find_max_gap(proc_ranges)

        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        steering_angle = self.get_angle(best, len(proc_ranges))
        self.vis.add_step(proc_ranges, steering_angle)

        max_steer = 0.4
        steering_angle = np.clip(steering_angle, -max_steer, max_steer)
        # steering_angle = steering_angle * 0.6

        return steering_angle

    def plan_act(self, obs):
        scan = obs[7:-1]
        ranges = np.array(scan, dtype=np.float)

        steering_angle = self.process_lidar(ranges)

        # speed = 4
        speed = calculate_speed(steering_angle)

        action = np.array([steering_angle, speed])

        return action

    def reset_lap(self):
        pass

#TODO: most of this can be njitted
class ForestFGM:    
    BUBBLE_RADIUS = 250
    PREPROCESS_CONV_SIZE = 3
    BEST_POINT_CONV_SIZE = 100
    MAX_LIDAR_DIST = 10
    REDUCTION = 200
    
    def __init__(self):
        # used when calculating the angles of the LiDAR data
        # self.vis = LidarViz(1000)
        self.degrees_per_elem = None
        self.name = "Follow the Forest Gap"
        self.n_beams = 1000
    
    def preprocess_lidar(self, ranges):
        self.degrees_per_elem = (180) / len(ranges)
        proc_ranges = np.array(ranges[self.REDUCTION:-self.REDUCTION])
        proc_ranges = ranges
        proc_ranges = np.clip(proc_ranges, 0, self.MAX_LIDAR_DIST)
        proc_ranges = np.convolve(proc_ranges, np.ones(self.PREPROCESS_CONV_SIZE), 'same') / self.PREPROCESS_CONV_SIZE

        return proc_ranges

   
    def find_best_point(self, start_i, end_i, ranges):
        """Start_i & end_i are start and end indices of max-gap range, respectively
        Return index of best point in ranges
	    Naive: Choose the furthest point within ranges and go there
        """
        # do a sliding window average over the data in the max gap, this will
        # help the car to avoid hitting corners
        averaged_max_gap = np.convolve(ranges[start_i:end_i], np.ones(self.BEST_POINT_CONV_SIZE), 'same') 
        averaged_max_gap = averaged_max_gap / self.BEST_POINT_CONV_SIZE
        best = averaged_max_gap.argmax()
        idx = best + start_i

        return idx

    def get_angle(self, range_index, range_len):
        """ Get the angle of a particular element in the LiDAR data
        """
        return (range_index - (range_len/2)) * self.degrees_per_elem 

    def process_lidar(self, ranges):
        proc_ranges = self.preprocess_lidar(ranges)
        closest = proc_ranges.argmin()

        min_index = closest - self.BUBBLE_RADIUS
        max_index = closest + self.BUBBLE_RADIUS
        if min_index < 0: min_index = 0
        if max_index >= len(proc_ranges): max_index = len(proc_ranges)-1
        proc_ranges[min_index:max_index] = 0

        gap_start, gap_end = find_max_gap(proc_ranges)

        best = self.find_best_point(gap_start, gap_end, proc_ranges)

        steering_angle = self.get_angle(best, len(proc_ranges))
        # self.vis.add_step(proc_ranges, steering_angle)

        return steering_angle

    def plan_act(self, obs):
        scan = obs[7:-1]
        ranges = np.array(scan, dtype=np.float)

        steering_angle = self.process_lidar(ranges)
        steering_angle = steering_angle * np.pi / 180

        # speed = 4
        speed = calculate_speed(steering_angle)

        action = np.array([steering_angle, speed])

        return action

    def reset_lap(self):
        pass

# @njit
def find_max_gap(free_space_ranges):
    """ Return the start index & end index of the max gap in free_space_ranges
        free_space_ranges: list of LiDAR data which contains a 'bubble' of zeros
    """
    # mask the bubble
    masked = np.ma.masked_where(free_space_ranges==0, free_space_ranges)
    # get a slice for each contigous sequence of non-bubble data
    slices = np.ma.notmasked_contiguous(masked)
    if len(slices) == 0:
        return 0, len(free_space_ranges)
    max_len = slices[0].stop - slices[0].start
    chosen_slice = slices[0]
    # I think we will only ever have a maximum of 2 slices but will handle an
    # indefinitely sized list for portablility
    for sl in slices[1:]:
        sl_len = sl.stop - sl.start
        if sl_len > max_len:
            max_len = sl_len
            chosen_slice = sl
    return chosen_slice.start, chosen_slice.stop


