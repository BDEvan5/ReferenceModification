import os
import shutil
from numba import njit
import numpy as np 
import csv
from matplotlib import pyplot as plt

from ReferenceModification.PlannerUtils import pure_pursuit_utils 
from ReferenceModification.PlannerUtils.TD3 import TD3 
import ReferenceModification.LibFunctions as lib 
from ReferenceModification.PlannerUtils.speed_utils import calculate_speed



class BaseMod:
    def __init__(self, agent_name, map_name, sim_conf) -> None:
        self.path_name = None

        self.wheelbase = sim_conf.l_f + sim_conf.l_r

        #TODO: move to config file
        self.v_gain = 0.5
        self.lookahead = 0.8
        self.max_reacquire = 20
        self.range_finder_scale = 5 #TODO: move to config files
        self.distance_scale = 20 # max meters for scaling

        self.waypoints = None
        self.vs = None

        self.name = agent_name
        self.n_beams = sim_conf.n_beams
        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer

        self._load_csv_track(map_name)

    def _get_current_waypoint(self, position):
        lookahead_distance = self.lookahead
    
        wpts = np.vstack((self.waypoints[:, 0], self.waypoints[:, 1])).T
        nearest_point, nearest_dist, t, i = pure_pursuit_utils.nearest_point_on_trajectory_py2(position, wpts)
        if nearest_dist < lookahead_distance:
            lookahead_point, i2, t2 = pure_pursuit_utils.first_point_on_trajectory_intersecting_circle(position, lookahead_distance, wpts, i+t, wrap=True)
            if i2 == None:
                return None
            current_waypoint = np.empty((3, ))
            # x, y
            current_waypoint[0:2] = wpts[i2, :]
            # speed
            current_waypoint[2] = self.vs[i]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], self.vs[i])
        else:
            return None

    def act_pp(self, pos, theta):
        lookahead_point = self._get_current_waypoint(pos)

        if lookahead_point is None:
            return [0, 4.0]

        speed, steering_angle = pure_pursuit_utils.get_actuation(theta, lookahead_point, pos, self.lookahead, self.wheelbase)

        speed = calculate_speed(steering_angle)

        return [steering_angle, speed]

    def transform_obs(self, obs, pp_action):
        """
        Transforms the observation received from the environment into a vector which can be used with a neural network.
    
        Args:
            obs: observation from env
            pp_action: [steer, speed] from pure pursuit controller

        Returns:
            nn_obs: observation vector for neural network
        """
        state = obs['state']
        cur_v = [state[3]/self.max_v]
        cur_d = [state[4]/self.max_steer]
        # vr_scale = [(pp_action[1])/self.max_v]
        target_angle = lib.get_bearing(state[0:2], [1, 21])
        angle = [lib.sub_angles_complex(target_angle, state[2])/self.max_steer]
        dr_scale = [pp_action[0]/self.max_steer]

        scan = np.array(obs['scan']) / self.range_finder_scale

        nn_obs = np.concatenate([cur_v, cur_d, angle, dr_scale, scan])

        return nn_obs

    def modify_references(self, nn_action, d_ref):
        """
        Modifies the reference quantities for the steering.
        Mutliplies the nn_action with the max steering and then sums with the reference

        Args:
            nn_action: action from neural network in range [-1, 1]
            d_ref: steering reference from PP

        Returns:
            d_new: modified steering reference
        """
        d_new = d_ref + self.max_steer * nn_action[0]
        d_new = np.clip(d_new, -self.max_steer, self.max_steer)

        return d_new

    def _load_csv_track(self, map_name):
        track = []
        filename = 'maps/' + map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        self.waypoints = track[:, 1:3]
        self.vs = track[:, 5]

        self.expand_wpts()

    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.waypoints
        o_vs = self.vs
        new_line = []
        new_vs = []
        for i in range(len(o_line)-1):
            dd = lib.sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = lib.add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        self.waypoints = np.array(new_line)
        self.vs = np.array(new_vs)


class ModVehicleTrain(BaseMod):
    def __init__(self, agent_name, map_name, sim_conf, load=False, h_size=200):
        """
        Training vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            sim_conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
            load: if the network should be loaded or recreated.
        """

        BaseMod.__init__(self, agent_name, map_name, sim_conf)

        self.path = 'Vehicles/' + agent_name
        state_space = 4 + self.n_beams
        self.agent = TD3(state_space, 1, 1, agent_name)
        h_size = h_size
        self.agent.try_load(load, h_size, self.path)

        self.observation = None
        self.nn_state = None
        self.nn_act = None
        self.action = None

        self.current_ep_reward = 0
        self.reward_ptr = 0
        self.ep_rewards = np.zeros(5000) # max 5000 eps
        self.step_counter = 0

        self.init_file_struct()

    def init_file_struct(self):
        path = os.getcwd() + '/' + self.path
        if os.path.exists(path):
            try:
                os.rmdir(path)
            except:
                shutil.rmtree(path)
        os.mkdir(path)
        
    def plan_act(self, obs):
        position = obs['state'][0:2]
        theta = obs['state'][2]
        pp_action = self.act_pp(position, theta)
        nn_obs = self.transform_obs(obs, pp_action)
        self.add_memory_entry(obs, nn_obs)

        self.observation = obs
        nn_action = self.agent.act(nn_obs)
        self.nn_act = nn_action
        self.nn_state = nn_obs
        self.step_counter += 1

        steering_angle = self.modify_references(self.nn_act, pp_action[0])

        speed = calculate_speed(steering_angle)
        self.action = np.array([steering_angle, speed])

        return self.action

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.observation is not None:
            reward = self.calculate_reward(s_prime)

            self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, False)

    def calculate_reward(self, s_prime):
        reward = s_prime['progress'] - self.observation['progress']
        reward += s_prime['reward']
        self.current_ep_reward += reward
        return reward

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        position = s_prime['state'][0:2]
        theta = s_prime['state'][2]
        pp_action = self.act_pp(position, theta)
        nn_s_prime = self.transform_obs(s_prime, pp_action)
        reward = self.calculate_reward(s_prime)

        self.ep_rewards[self.reward_ptr] = self.current_ep_reward
        self.current_ep_reward = 0 # reset
        self.reward_ptr += 1
        if self.reward_ptr % 10 == 0:
            self.print_update(True)
            self.agent.save(self.path)
        self.observation = None

        self.agent.replay_buffer.add(self.nn_state, self.nn_act, nn_s_prime, reward, True)

    def print_update(self, plot_reward=True):
        if self.reward_ptr < 5:
            return
        mean = np.mean(self.ep_rewards[max(0, self.reward_ptr-101):self.reward_ptr-1])
        print(f"Run: {self.step_counter} --> 100 ep Mean: {mean:.2f}  ")
        
        if plot_reward:
            lib.plot(self.ep_rewards[0:self.reward_ptr], 20, figure_n=2)

    def save_csv_data(self):
        data = []
        for i in range(len(self.ep_rewards)):
            data.append([i, self.ep_rewards[i]])
        full_name = self.path + '/training_data.csv'
        with open(full_name, 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerows(data)

        plt.figure(2)
        plt.savefig(self.path + "/training_rewards.png")


class ModVehicleTest(BaseMod):
    def __init__(self, agent_name, map_name, sim_conf):
        """
        Testing vehicle using the reference modification navigation stack

        Args:
            agent_name: name of the agent for saving and reference
            sim_conf: namespace with simulation parameters
            mod_conf: namespace with modification planner parameters
        """

        BaseMod.__init__(self, agent_name, map_name, sim_conf)

        self.path = 'Vehicles/' + agent_name
        state_space = 4 + self.n_beams
        self.agent = TD3(state_space, 1, 1, agent_name)
        self.agent.load(directory=self.path)
        self.n_beams = 10

        print(f"Agent loaded: {agent_name}")

        # self.vis = LidarVizMod(10)

    def plan_act(self, obs):
        position = obs['state'][0:2]
        theta = obs['state'][2]
        pp_action = self.act_pp(position, theta)
        nn_obs = self.transform_obs(obs, pp_action)

        nn_action = self.agent.act(nn_obs, noise=0)
        self.nn_act = nn_action

        steering_angle = self.modify_references(self.nn_act, pp_action[0])
        speed = calculate_speed(steering_angle)
        action = np.array([steering_angle, speed])

        return action
