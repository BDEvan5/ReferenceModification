import numpy as np 
from matplotlib import pyplot as plt
import os

import ReferenceModification.LibFunctions as lib
from ReferenceModification.Simulator.SimMaps import TrackMap, ForestMap

from ReferenceModification.Simulator.BaseSimClasses import CarModel, ScanSimulator, BaseSim, SimHistory 


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
        # horizontal_force = self.car.mass * self.car.th_dot * self.car.velocity
        # self.y_forces.append(horizontal_force)
        # if horizontal_force > self.car.max_friction_force:
            # self.done = True
            # self.reward = -1
            # self.done_reason = f"Friction limit reached: {horizontal_force} > {self.car.max_friction_force}"
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
            path = os.path.dirname(__file__)
            sim_conf = lib.load_conf(path, "std_config")

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
        dx = self.car.x - self.env_map.start_pose[0]
        dx_lim = self.env_map.forest_width * 0.5
        if dx < dx_lim and self.car.y > self.env_map.end_y:
            self.done = True
            self.reward = 1
            self.done_reason = f"Lap complete"

        # check crash
        elif self.env_map.check_scan_location([self.car.x, self.car.y]):
            self.done = True
            self.reward = -1
            self.done_reason = f"Crash obstacle: [{self.car.x:.2f}, {self.car.y:.2f}]"
        # horizontal_force = self.car.mass * self.car.th_dot * self.car.velocity
        # check forces
        # if horizontal_force > self.car.max_friction_force:
            # self.done = True
            # self.reward = -1
            # print(f"ThDot: {self.car.th_dot} --> Vel: {self.car.velocity}")
            # self.done_reason = f"Friction: {horizontal_force} > {self.car.max_friction_force}"

        # check steps
        elif self.steps > self.max_steps:
            self.done = True
            self.reward = -1
            self.done_reason = f"Max steps"
        # check orientation
        elif abs(self.car.theta) > 0.66*np.pi:
            self.done = True
            self.done_reason = f"Vehicle turned around"
            self.reward = -1

        return self.done


    

