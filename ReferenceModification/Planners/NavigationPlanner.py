import numpy as np 
import csv
from matplotlib import pyplot as plt
from numba import njit

from ReferenceModification.PlannerUtils.TD3 import TD3 
from ReferenceModification.PlannerUtils.speed_utils import calculate_speed
from ReferenceModification.PlannerUtils.speed_utils import calculate_speed

import ReferenceModification.LibFunctions as lib

import os
import shutil

class BaseNav:
    def __init__(self, agent_name, sim_conf) -> None:
        self.name = agent_name
        self.n_beams = sim_conf.n_beams
        self.max_v = sim_conf.max_v
        self.max_steer = sim_conf.max_steer

        self.distance_scale = 20 # max meters for scaling
        self.range_finder_scale = 5

    def transform_obs(self, obs):
        observation = obs['state']
        cur_v = [observation[3]/self.max_v]
        cur_d = [observation[4]/self.max_steer]
        angle = [lib.get_bearing(observation[0:2], [1, 21])/self.max_steer]
        scan = np.array(obs['scan']) / self.range_finder_scale

        nn_obs = np.concatenate([cur_v, cur_d, angle, scan])

        return nn_obs

    
class NavTrainVehicle(BaseNav):
    def __init__(self, agent_name, sim_conf, load=False, h_size=200) -> None:
        BaseNav.__init__(self, agent_name, sim_conf)
        self.path = 'Vehicles/' + agent_name
        observation_space = 3 + self.n_beams
        self.agent = TD3(observation_space, 1, 1, agent_name)
        self.agent.try_load(load, h_size, self.path)

        self.observation = None
        self.action = None
        self.nn_state = None
        self.nn_action = None

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
        self.step_counter += 1
        nn_obs = self.transform_obs(obs)
        self.add_memory_entry(obs, nn_obs)

        nn_action = self.agent.act(nn_obs)
        
        self.observation = obs
        self.nn_state = nn_obs
        self.nn_action = nn_action

        steering_angle = nn_action[0] * self.max_steer
        speed = calculate_speed(steering_angle)
        self.action = np.array([steering_angle, speed])
        
        return self.action

    def calculate_reward(self, s_prime):
        reward = s_prime['progress'] - self.observation['progress']
        reward += s_prime['reward']
        self.current_ep_reward += reward

        return reward

    def add_memory_entry(self, s_prime, nn_s_prime):
        if self.observation is not None:
            reward = self.calculate_reward(s_prime)

            self.agent.replay_buffer.add(self.nn_state, self.nn_action, nn_s_prime, reward, False)

    def done_entry(self, s_prime):
        """
        To be called when ep is done.
        """
        nn_s_prime = self.transform_obs(s_prime)
        reward = self.calculate_reward(s_prime)

        self.ep_rewards[self.reward_ptr] = self.current_ep_reward
        self.current_ep_reward = 0 # reset
        self.reward_ptr += 1
        if self.reward_ptr % 10 == 0:
            self.print_update(True)
            self.agent.save(self.path)
        self.observation = None

        self.agent.replay_buffer.add(self.nn_state, self.nn_action, nn_s_prime, reward, True)

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



class NavTestVehicle(BaseNav):
    def __init__(self, agent_name, sim_conf) -> None:
        BaseNav.__init__(self, agent_name, sim_conf)
        self.path = 'Vehicles/' + agent_name
        observation_space = 2 + self.n_beams
        self.agent = TD3(observation_space, 1, 1, agent_name)
        h_size = 200
        self.agent.try_load(True, h_size, self.path)
        self.n_beams = 10

    def plan_act(self, obs):
        nn_obs = self.transform_obs(obs)
        nn_action = self.agent.act(nn_obs)
        steering_angle = self.max_steer * nn_action[0]

        speed = calculate_speed(steering_angle)
        action = np.array([steering_angle, speed])

        return action

    def reset_lap(self):
        pass

