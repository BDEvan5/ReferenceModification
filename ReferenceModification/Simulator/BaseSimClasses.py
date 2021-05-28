import numpy as np 
from matplotlib import pyplot as plt
import os
from numba import njit

import ReferenceModification.LibFunctions as lib


class CarModel:
    """
    A simple class which holds the state of a car and can update the dynamics based on the bicycle model

    Data Members:
        x: x location of vehicle on map
        y: y location of vehicle on map
        theta: orientation of vehicle
        velocity: 
        steering: delta steering angle
        th_dot: the change in orientation due to steering

    """
    def __init__(self, sim_conf):
        """
        Init function

        Args:
            sim_conf: a config namespace with relevant car parameters
        """
        self.x = 0
        self.y = 0
        self.theta = 0
        self.velocity = 0
        self.steering = 0
        self.th_dot = 0

        self.prev_loc = 0

        self.wheelbase = sim_conf.l_f + sim_conf.l_r
        self.mass = sim_conf.m
        self.mu = sim_conf.mu

        self.max_d_dot = sim_conf.max_d_dot
        self.max_steer = sim_conf.max_steer
        self.max_a = sim_conf.max_a
        self.max_v = sim_conf.max_v
        self.max_friction_force = self.mass * self.mu * 9.81

    def update_kinematic_state(self, a, d_dot, dt):
        """
        Updates the internal state of the vehicle according to the kinematic equations for a bicycle model

        Args:
            a: acceleration
            d_dot: rate of change of steering angle
            dt: timestep in seconds

        """
        self.x = self.x + self.velocity * np.sin(self.theta) * dt
        self.y = self.y + self.velocity * np.cos(self.theta) * dt
        theta_dot = self.velocity / self.wheelbase * np.tan(self.steering)
        self.th_dot = theta_dot
        dth = theta_dot * dt
        self.theta = lib.add_angles_complex(self.theta, dth)

        a = np.clip(a, -self.max_a, self.max_a)
        d_dot = np.clip(d_dot, -self.max_d_dot, self.max_d_dot)

        self.steering = self.steering + d_dot * dt
        self.velocity = self.velocity + a * dt

        self.steering = np.clip(self.steering, -self.max_steer, self.max_steer)
        self.velocity = np.clip(self.velocity, -self.max_v, self.max_v)

    def get_car_state(self):
        """
        Returns the state of the vehicle as an array

        Returns:
            state: [x, y, theta, velocity, steering]

        """
        state = []
        state.append(self.x) #0
        state.append(self.y)
        state.append(self.theta) # 2
        state.append(self.velocity) #3
        state.append(self.steering)  #4

        state = np.array(state)

        return state

    def reset_state(self, start_pose):
        """
        Resets the state of the vehicle

        Args:
            start_pose: the starting, [x, y, theta] to reset to
        """
        self.x = start_pose[0]
        self.y = start_pose[1]
        self.theta = start_pose[2]
        self.velocity = 0
        self.steering = 0
        self.prev_loc = [self.x, self.y]


class ScanSimulator:
    def __init__(self, number_of_beams=10, fov=np.pi, std_noise=0.01):
        self.number_of_beams = number_of_beams
        self.fov = fov 
        self.std_noise = std_noise
        self.rng = np.random.default_rng(seed=12345)

        self.dth = self.fov / (self.number_of_beams -1)

        self.map_height = None
        self.map_width = None
        self.resoltuion = None
        self.orig_x = 0
        self.orig_y = 0

        self.eps = 0.01
        self.max_range = 10

    def reset_n_beams(self, n_beams):
        self.number_of_beams = n_beams
        self.dth = self.fov / (self.number_of_beams -1)


    def init_sim_map(self, env_map):
        self.map_height = env_map.map_height
        self.map_width = env_map.map_width
        self.resoltuion = env_map.resolution
        self.orig_x = env_map.origin[0]
        self.orig_y = env_map.origin[1]

        self.dt = env_map.dt_img

    def scan(self, pose):
        scan = get_scan(pose, self.number_of_beams, self.dth, self.dt, self.fov, self.orig_x, self.orig_y, self.resoltuion, self.map_height, self.map_width, self.eps, self.max_range)

        noise = self.rng.normal(0., self.std_noise, size=self.number_of_beams)
        final_scan = scan + noise
        return final_scan


@njit(cache=True)
def get_scan(pose, num_beams, th_increment, dt, fov, orig_x, orig_y, resolution, height, width, eps, max_range):
    x = pose[0]
    y = pose[1]
    theta = pose[2]

    scan = np.empty(num_beams)

    for i in range(num_beams):
        scan_theta = theta + th_increment * i - fov/2
        scan[i] = trace_ray(x, y, scan_theta, dt, orig_x, orig_y, resolution, height, width, eps, max_range)

    return scan


@njit(cache=True)
def trace_ray(x, y, theta, dt, orig_x, orig_y, resolution, height, width, eps, max_range):
    s = np.sin(theta)
    c = np.cos(theta)

    distance_to_nearest = get_distance_dt(x, y, dt, orig_x, orig_y, resolution, height, width)
    total_distance = distance_to_nearest

    while distance_to_nearest > eps and total_distance <= max_range:
        x += distance_to_nearest * s
        y += distance_to_nearest * c


        distance_to_nearest = get_distance_dt(x, y, dt, orig_x, orig_y, resolution, height, width)
        total_distance += distance_to_nearest

    if total_distance > max_range:
        total_distance = max_range

    return total_distance

@njit(cache=True)
def get_distance_dt(x, y, dt, orig_x, orig_y, resolution, height, width):

    c = int((x -orig_x) / resolution)
    r = int((y-orig_y) / resolution)

    if c >= width or r >= height:
        return 0

    distance = dt[r, c]

    return distance


#TODO: move this to another location
class SimHistory:
    def __init__(self, sim_conf):
        self.sim_conf = sim_conf
        self.positions = []
        self.steering = []
        self.velocities = []
        self.obs_locations = []
        self.thetas = []


        self.ctr = 0

    def save_history(self):
        pos = np.array(self.positions)
        vel = np.array(self.velocities)
        steer = np.array(self.steering)
        obs = np.array(self.obs_locations)

        d = np.concatenate([pos, vel[:, None], steer[:, None]], axis=-1)

        d_name = 'Vehicles/TrainData/' + f'data{self.ctr}'
        o_name = 'Vehicles/TrainData/' + f"obs{self.ctr}"
        np.save(d_name, d)
        np.save(o_name, obs)

    def reset_history(self):
        self.positions = []
        self.steering = []
        self.velocities = []
        self.obs_locations = []
        self.thetas = []

        self.ctr += 1

    def show_history(self, vs=None):
        plt.figure(1)
        plt.clf()
        plt.title("Steer history")
        plt.plot(self.steering)
        plt.pause(0.001)

        plt.figure(2)
        plt.clf()
        plt.title("Velocity history")
        plt.plot(self.velocities)
        if vs is not None:
            r = len(vs) / len(self.velocities)
            new_vs = []
            for i in range(len(self.velocities)):
                new_vs.append(vs[int(round(r*i))])
            plt.plot(new_vs)
            plt.legend(['Actual', 'Planned'])
        plt.pause(0.001)

    def show_forces(self):
        mu = self.sim_conf['car']['mu']
        m = self.sim_conf['car']['m']
        g = self.sim_conf['car']['g']
        l_f = self.sim_conf['car']['l_f']
        l_r = self.sim_conf['car']['l_r']
        f_max = mu * m * g
        f_long_max = l_f / (l_r + l_f) * f_max

        self.velocities = np.array(self.velocities)
        self.thetas = np.array(self.thetas)

        # divide by time taken for change to get per second
        t = self.sim_conf['sim']['timestep'] * self.sim_conf['sim']['update_f']
        v_dot = (self.velocities[1:] - self.velocities[:-1]) / t
        oms = (self.thetas[1:] - self.thetas[:-1]) / t

        f_lat = oms * self.velocities[:-1] * m
        f_long = v_dot * m
        f_total = (f_lat**2 + f_long**2)**0.5

        plt.figure(3)
        plt.clf()
        plt.title("Forces (lat, long)")
        plt.plot(f_lat)
        plt.plot(f_long)
        plt.plot(f_total, linewidth=2)
        plt.legend(['Lat', 'Long', 'total'])
        plt.plot(np.ones_like(f_lat) * f_max, '--')
        plt.plot(np.ones_like(f_lat) * f_long_max, '--')
        plt.plot(-np.ones_like(f_lat) * f_max, '--')
        plt.plot(-np.ones_like(f_lat) * f_long_max, '--')
        plt.pause(0.001)


class BaseSim:
    """
    Base simulator class

    Important parameters:
        timestep: how long the simulation steps for
        max_steps: the maximum amount of steps the sim can take

    Data members:
        car: a model of a car with the ability to update the dynamics
        scan_sim: a simulator for a laser scanner
        action: the current action which has been given
        history: a data logger for the history
    """
    def __init__(self, env_map, done_fcn, sim_conf):
        """
        Init function

        Args:
            env_map: an env_map object which holds a map and has mapping functions
            done_fcn: a function which checks the state of the simulation for episode completeness
        """
        self.done_fcn = done_fcn
        self.env_map = env_map
        self.sim_conf = sim_conf #TODO: don't store the conf file, just use and throw away.
        self.n_obs = self.env_map.n_obs

        self.timestep = self.sim_conf.time_step
        self.max_steps = self.sim_conf.max_steps
        self.plan_steps = self.sim_conf.plan_steps

        self.car = CarModel(self.sim_conf)
        self.scan_sim = ScanSimulator(self.sim_conf.n_beams)
        self.scan_sim.init_sim_map(env_map)
        # self.scan_sim.set_check_fcn(self.env_map.check_scan_location)

        self.done = False
        self.colission = False
        self.reward = 0
        self.action = np.zeros((2))
        self.action_memory = []
        self.steps = 0

        self.history = SimHistory(self.sim_conf)
        self.done_reason = ""

    def step_control(self, action):
        """
        Steps the simulator for a single step

        Args:
            action: [steer, speed]
        """
        d_ref = np.clip(action[0], -self.car.max_steer, self.car.max_steer)
        v_ref = np.clip(action[1], 0, self.car.max_v)
        acceleration, steer_dot = self.control_system(v_ref, d_ref)
        self.car.update_kinematic_state(acceleration, steer_dot, self.timestep)
        self.steps += 1

        return self.done_fcn()

    def step_plan(self, action):
        """
        Takes multiple control steps based on the number of control steps per planning step

        Args:
            action: [steering, speed]
            done_fcn: a no arg function which checks if the simulation is complete
        """

        for _ in range(self.plan_steps):
            if self.step_control(action):
                break

        self.record_history(action)

        obs = self.get_observation()
        done = self.done
        reward = self.reward

        return obs, reward, done, None

    def record_history(self, action):
        self.action = action
        self.history.velocities.append(self.car.velocity)
        self.history.steering.append(self.car.steering)
        self.history.positions.append([self.car.x, self.car.y])
        self.history.thetas.append(self.car.theta)

    def control_system(self, v_ref, d_ref):
        """
        Generates acceleration and steering velocity commands to follow a reference
        Note: the controller gains are hand tuned in the fcn

        Args:
            v_ref: the reference velocity to be followed
            d_ref: reference steering to be followed

        Returns:
            a: acceleration
            d_dot: the change in delta = steering velocity
        """

        kp_a = 10
        a = (v_ref - self.car.velocity) * kp_a
        
        kp_delta = 40
        d_dot = (d_ref - self.car.steering) * kp_delta

        a = np.clip(a, -8, 8)
        d_dot = np.clip(d_dot, -3.2, 3.2)

        return a, d_dot

    def reset(self, add_obs=True):
        """
        Resets the simulation

        Args:
            add_obs: a boolean flag if obstacles should be added to the map

        Returns:
            state observation
        """
        self.done = False
        self.done_reason = "Null"
        self.action_memory = []
        self.steps = 0
        self.reward = 0

        #TODO: move this reset to inside car
        self.car.reset_state(self.env_map.start_pose)


        self.history.reset_history()

        if add_obs:
            self.env_map.add_obstacles()

        # update the dt img in the scan simulator after obstacles have been added
        dt = self.env_map.set_dt()
        self.scan_sim.dt = dt

        return self.get_observation()

    def render(self, wait=False, name="No vehicle name set"):
        """
        Renders the map using the plt library

        Args:
            wait: plt.show() should be called or not
        """
        self.env_map.render_map(4)
        # plt.show()
        fig = plt.figure(4)
        plt.title(name)

        xs, ys = self.env_map.convert_positions(self.history.positions)
        plt.plot(xs, ys, 'r', linewidth=3)
        plt.plot(xs, ys, '+', markersize=12)

        x, y = self.env_map.xy_to_row_column([self.car.x, self.car.y])
        plt.plot(x, y, 'x', markersize=20)

        text_x = self.env_map.map_width + 1
        text_y = self.env_map.map_height / 10

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(text_x, text_y * 1, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(text_x, text_y * 2, s) 
        s = f"Done: {self.done}"
        plt.text(text_x, text_y * 3, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(text_x, text_y * 4, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(text_x, text_y * 5, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(text_x, text_y * 6, s) 
        s = f"Delta x100: [{(self.car.steering*100):.2f}]"
        plt.text(text_x, text_y * 7, s) 
        s = f"Done reason: {self.done_reason}"
        plt.text(text_x, text_y * 8, s) 
        

        s = f"Steps: {self.steps}"
        plt.text(text_x, text_y * 9, s)


        plt.pause(0.0001)
        if wait:
            plt.show()

    def min_render(self, wait=False):
        """
        TODO: deprecate
        """
        fig = plt.figure(4)
        plt.clf()  

        ret_map = self.env_map.scan_map
        plt.imshow(ret_map)

        # plt.xlim([0, self.env_map.width])
        # plt.ylim([0, self.env_map.height])

        s_x, s_y = self.env_map.convert_to_plot(self.env_map.start)
        plt.plot(s_x, s_y, '*', markersize=12)

        c_x, c_y = self.env_map.convert_to_plot([self.car.x, self.car.y])
        plt.plot(c_x, c_y, '+', markersize=16)

        for i in range(self.scan_sim.number_of_beams):
            angle = i * self.scan_sim.dth + self.car.theta - self.scan_sim.fov/2
            fs = self.scan_sim.scan_output[i] * self.scan_sim.n_searches * self.scan_sim.step_size
            dx =  [np.sin(angle) * fs, np.cos(angle) * fs]
            range_val = lib.add_locations([self.car.x, self.car.y], dx)
            r_x, r_y = self.env_map.convert_to_plot(range_val)
            x = [c_x, r_x]
            y = [c_y, r_y]

            plt.plot(x, y)

        for pos in self.action_memory:
            p_x, p_y = self.env_map.convert_to_plot(pos)
            plt.plot(p_x, p_y, 'x', markersize=6)

        text_start = self.env_map.width + 10
        spacing = int(self.env_map.height / 10)

        s = f"Reward: [{self.reward:.1f}]" 
        plt.text(text_start, spacing*1, s)
        s = f"Action: [{self.action[0]:.2f}, {self.action[1]:.2f}]"
        plt.text(text_start, spacing*2, s) 
        s = f"Done: {self.done}"
        plt.text(text_start, spacing*3, s) 
        s = f"Pos: [{self.car.x:.2f}, {self.car.y:.2f}]"
        plt.text(text_start, spacing*4, s)
        s = f"Vel: [{self.car.velocity:.2f}]"
        plt.text(text_start, spacing*5, s)
        s = f"Theta: [{(self.car.theta * 180 / np.pi):.2f}]"
        plt.text(text_start, spacing*6, s) 
        s = f"Delta x100: [{(self.car.steering*100):.2f}]"
        plt.text(text_start, spacing*7, s) 
        s = f"Theta Dot: [{(self.car.th_dot):.2f}]"
        plt.text(text_start, spacing*8, s) 

        s = f"Steps: {self.steps}"
        plt.text(100, spacing*9, s)

        plt.pause(0.0001)
        if wait:
            plt.show()
    
    def get_target_obs(self):
        target = self.env_map.end_goal
        pos = np.array([self.car.x, self.car.y])
        base_angle = lib.get_bearing(pos, target) 
        # angle = base_angle - self.car.theta
        angle = lib.sub_angles_complex(base_angle, self.car.theta)
        # angle = lib.add_angles_complex(base_angle, self.car.theta)
        # distance = lib.get_distance(pos, target)

        em = self.env_map
        s = calculate_progress(pos, em.ref_pts, em.diffs, em.l2s, em.ss_normal)

        return [angle, s]
    
    def get_observation(self):
        """
        Combines different parts of the simulator to get a state observation which can be returned.
        """
        car_obs = self.car.get_car_state()
        pose = car_obs[0:3]
        scan = self.scan_sim.scan(pose)
        target = self.get_target_obs()

        observation = np.concatenate([car_obs, target, scan, [self.reward]])
        return observation

@njit(cache=True)
def calculate_progress(point, wpts, diffs, l2s, ss):
    dots = np.empty((wpts.shape[0]-1, ))
    dots_shape = dots.shape[0]
    for i in range(dots_shape):
        dots[i] = np.dot((point - wpts[i, :]), diffs[i, :])
    t = dots / l2s
    t[t<0.0] = 0.0
    t[t>1.0] = 1.0  #np.clip, unsupported
    
    projections = wpts[:-1,:] + (t*diffs.T).T

    # dists = np.linalg.norm(point - projections, axis=1)
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp*temp))

    min_dist_segment = np.argmin(dists)
    dist_from_cur_pt = dists[min_dist_segment]

    s = ss[min_dist_segment] + dist_from_cur_pt
    # print(F"{min_dist_segment} --> SS: {ss[min_dist_segment]}, curr_pt: {dist_from_cur_pt}")

    s = s / ss[-1]

    return s 


