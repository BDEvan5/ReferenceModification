import numpy as np
from numba import njit
from matplotlib import pyplot as plt
import csv

from ReferenceModification.PlannerUtils.TrajectoryPlanner import MinCurvatureTrajectory

import ReferenceModification.LibFunctions as lib

from ReferenceModification.PlannerUtils.speed_utils import calculate_speed
from ReferenceModification.PlannerUtils import pure_pursuit_utils


class OraclePP:
    def __init__(self, sim_conf) -> None:
        self.name = "Oracle Path Follower"

        self.wheelbase = sim_conf.l_f + sim_conf.l_r

        self.v_gain = 0.5
        self.lookahead = 0.8
        self.max_reacquire = 20

        self.waypoints = None
        self.vs = None

        self.aim_pts = []

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
            current_waypoint[2] = self.waypoints[i, 2]
            return current_waypoint
        elif nearest_dist < self.max_reacquire:
            return np.append(wpts[i, :], self.waypoints[i, 2])
        else:
            return None

    def act_pp(self, pos, theta):
        lookahead_point = self._get_current_waypoint(pos)

        if lookahead_point is None:
            return [0, 4.0]

        speed, steering_angle = pure_pursuit_utils.get_actuation(theta, lookahead_point, pos, self.lookahead, self.wheelbase)

        # speed = 4
        speed = calculate_speed(steering_angle)

        return [steering_angle, speed]

    def reset_lap(self):
        self.aim_pts.clear()


class Oracle(OraclePP):
    def __init__(self, sim_conf):
        OraclePP.__init__(self, sim_conf)
        self.sim_conf = sim_conf # kept for optimisation
        self.n_beams = 10

    def plan_track(self, env_map):
        track = []
        filename = 'maps/' + env_map.map_name + "_opti.csv"
        with open(filename, 'r') as csvfile:
            csvFile = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC)  
        
            for lines in csvFile:  
                track.append(lines)

        track = np.array(track)
        print(f"Track Loaded: {filename}")

        wpts = track[:, 1:3]
        vs = track[:, 5]

        self.waypoints = np.concatenate([wpts, vs[:, None]], axis=-1)
        self.expand_wpts()

        return self.waypoints[:, 0:2]

    def expand_wpts(self):
        n = 5 # number of pts per orig pt
        dz = 1 / n
        o_line = self.waypoints[:, 0:2]
        # o_ss = self.ss
        o_vs = self.waypoints[:, 2]
        new_line = []
        # new_ss = []
        new_vs = []
        for i in range(len(self.waypoints)-1):
            dd = lib.sub_locations(o_line[i+1], o_line[i])
            for j in range(n):
                pt = lib.add_locations(o_line[i], dd, dz*j)
                new_line.append(pt)

                # ds = o_ss[i+1] - o_ss[i]
                # new_ss.append(o_ss[i] + dz*j*ds)

                dv = o_vs[i+1] - o_vs[i]
                new_vs.append(o_vs[i] + dv * j * dz)

        wpts = np.array(new_line)
        # self.ss = np.array(new_ss)
        vs = np.array(new_vs)
        self.waypoints = np.concatenate([wpts, vs[:, None]], axis=-1)

    def plan_forest(self, env_map):
        # load center pts
        start_x = env_map.start_pose[0]
        start_y = env_map.start_pose[1]
        max_width = env_map.start_pose[0] * 0.75
        length = env_map.end_y - start_y
        num_pts = 50

        y_pts = np.linspace(start_y, length+start_y, num_pts)[:, None]
        t_pts = np.concatenate([np.ones_like(y_pts)*start_x, y_pts], axis=-1)
        
        ws = find_true_widths2(t_pts, max_width, env_map.check_plan_location)

        # Optimise n_set
        N = len(t_pts)
        nvecs = np.concatenate([np.ones((N, 1)), np.zeros((N, 1))], axis=-1)
        n_set = MinCurvatureTrajectory(t_pts, nvecs, ws)

        waypoints = np.concatenate([np.ones((N, 1))*start_x + n_set, y_pts], axis=-1)
        velocity = 1
        vs = np.ones((N, 1)) * velocity

        self.waypoints = np.concatenate([waypoints, vs], axis=-1)

        # self.plot_plan(env_map, t_pts, ws, waypoints)

        self.reset_lap()

        return waypoints

    def plot_plan(self, env_map, t_pts, ws, waypoints=None):
        env_map.render_map(4)

        plt.figure(4)
        env_map.render_wpts(t_pts)
        env_map.render_wpts(waypoints)
        # env_map.render_aim_pts(t_pts)

        rs = t_pts[:, 0] - ws[:, 0]
        r_pts = np.stack([rs, t_pts[:, 1]], axis=1)
        xs, ys = env_map.convert_positions(r_pts)
        plt.plot(xs, ys, 'b')

        ls = t_pts[:, 0] + ws[:, 1]
        l_pts = np.stack([ls, t_pts[:, 1]], axis=1)
        xs, ys = env_map.convert_positions(l_pts)
        plt.plot(xs, ys, 'b')

        plt.show()

    def plan_act(self, obs):
        pos = np.array(obs['state'][0:2])
        theta = obs['state'][2]
        return self.act_pp(pos, theta)
        


# @njit
#TODO: pass dt to it so that I can njit it.
def find_true_widths2(t_pts, max_width, check_scan_location):
    tx = t_pts[:, 0]
    ty = t_pts[:, 1]

    stp_sze = 0.1
    N = len(t_pts)
    ws = np.zeros((N, 2))
    for i in range(N):
        pt = np.array([tx[i], ty[i]])

        #TODO: update the check to use the dt value. ame as lidar scan
        if not check_scan_location(pt):
            j = stp_sze
            s_pt = pt + [j, 0]
            while not check_scan_location(s_pt) and j < max_width:
                j += stp_sze
                s_pt = pt + [j, 0]
            ws[i, 1] = j 

            j = stp_sze
            s_pt = pt - np.array([j, 0])
            while not check_scan_location(s_pt) and j < max_width:
                j += stp_sze
                s_pt = pt - np.array([j, 0])
            ws[i, 0] = j
        else:
            for j in np.linspace(0, max_width, 10):
                p_pt = pt + [j, 0]
                n_pt = pt - [j, 0]
                if not check_scan_location(p_pt):
                    ws[i, 0] = -j 
                    ws[i, 1] = max_width
                    break
                elif not check_scan_location(n_pt):
                    ws[i, 0] = max_width
                    ws[i, 1] = -j
                    break 

    return ws

