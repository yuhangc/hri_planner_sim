import numpy as np
import matplotlib.pyplot as plt

from utils import *


class LocalPlannerBase:
    def __init__(self, dt, horizon, occupancy=None, global_planner=None, dyn_obstacles=None):
        """
        Initialization of the basic local planner
        :param dt: discrete time step size
        :param horizon: number of time steps to plan
        :param occupancy: occupancy grid map object
        :param global_planner: plan/path from the global planner
        :param dyn_obstacles: list of real-time detected obstacles
        """
        self.dt = dt
        self.horizon = horizon
        self.occupancy = occupancy
        self.global_planner = global_planner
        self.dyn_obstacles = dyn_obstacles

        # flag indicates whether there is a local plan
        self.has_plan = False

        # stores the local plan
        self.actions = None

    def update_occupancy(self, occupancy_new):
        self.occupancy = occupancy_new

    def update_global_planner(self, planner_new):
        self.global_planner = planner_new

    def update_obstacles(self, obstacles):
        self.dyn_obstacles = obstacles

    def get_action(self, all_actions=False):
        """
        Get action from the local plan, by default only return the first action
        unless "all_actions" is set to True
        """
        if all_actions:
            return self.actions
        else:
            return self.actions[0]

    def solve(self, x_curr, vel_curr):
        """
        Local planner solves for plan starting from x_curr
        Need to be implemented by subclasses
        :param x_curr: current state
        :param vel_curr: current velocity
        :return: True if solved, False if unable to get a plan
        """
        raise Exception("solve method must be overriden by subclasses!")

    def visualize_plan(self):
        """
        Visualize the local plan, need to be implemented by subclasses
        """
        raise Exception("visualize plan method must be overriden by subclasses!")


class DynamicWindowPlanner(LocalPlannerBase):
    def __init__(self, dt, horizon, occupancy=None, global_plan=None, dyn_obstacles=None,
                 robot_radius=None, config_file_path=None):
        """
        Initialize the dynamic window motion planner
        :param config_file_path: file path to the planner configuration file
        """
        LocalPlannerBase.__init__(self, dt, horizon, occupancy, global_plan, dyn_obstacles)
        self.robot_radius = robot_radius

        if config_file_path is None:
            # set maximum velocity and acc
            self.vel_max = np.array([0.5, 1.5])
            self.acc_max = np.array([0.5, 1.0])
            self.window_size = self.acc_max * self.dt

            # set weights for cost functions
            self.w_costs = {'heading': 2.0,
                            'collision': 0.2,
                            'velocity': 0.2}

            # velocity discretization
            self.vel_inc = np.array([0.05, 0.1])

            # max obstacle distance to be considered
            self.dist_obs_max = 5

            # max cost (for non-admissible velocities)
            self.cost_max = 1e6

            # clearance coefficient for collision checking
            self.clear_coeff = 0.1

            # whether to interpolate for global plan
            self.flag_interpolate = True

            # estimated planner running time
            self.dt_plan = 0.1
        else:
            # load from file
            self.load_configure(config_file_path)

        # direction to angle for global plan
        self.dir_to_ang = np.pi/4.0 * np.array([0, -2.0, 2.0, 4.0, -1.0, 1.0, -3.0, 3.0])

    def load_configure(self, config_file_path):
        pass

    def set_robot_radius(self, radius):
        self.robot_radius = radius

    def collision_check(self, x, radius, obs, clearance=None):
        """
        Collision check for given robot footprint radius and obstacle
        :param radius: assuming circular robot footprint
        :param obs: assuming circular obstacles
        :param clearance: a single float number
        """
        pass

    def update_obstacle_dists(self, x, radius, v, omg, clearance=None):
        """
        Calculate minimum distance of the obstacles to the robot along the
        circular path. If no intersection, dist is self.dist_obs_max
        :param x: current robot position
        :param radius: robot radius, assuming circular robot
        :param v: desired linear speed
        :param omg: desired angular speed
        :param clearance: clearance for collision detection
        :return: array of obstacle distances
        """
        dist_obs = np.zeros((len(self.dyn_obstacles),))
        idx = -1

        th = x[2]
        if omg > TOL:
            # calculate circular path parameters
            r = v / omg
            r_min = np.abs(r) - radius - clearance
            r_max = np.abs(r) + radius + clearance
            cx = x[0] - r * np.sin(th)
            cy = x[1] + r * np.cos(th)
            for obs in self.dyn_obstacles:
                idx += 1
                r_hat = distance((obs.x[0], obs.x[1]), (cx, cy))
                if r_min - obs.radius < r_hat < r_max + obs.radius:
                    # if obstacle intersects path
                    phi = np.arctan2(obs.x[1] - cy, obs.x[0] - cx) - th - np.pi / 2.0
                    if phi * omg > 0:
                        # if in the correct direction
                        dist_obs[idx] = np.abs(r * phi)
                    else:
                        dist_obs[idx] = self.dist_obs_max
                else:
                    dist_obs[idx] = self.dist_obs_max

        else:
            # calcualte straight path parameters
            for obs in self.dyn_obstacles:
                idx += 1
                phi = np.arctan2(obs.x[1] - x[1], obs.x[0] - x[0])

                if np.cos(th - phi) * v < 0:
                    # opposite direction
                    dist_obs[idx] = self.dist_obs_max
                else:
                    d = distance(x[0:2], obs.x)
                    d_hat = d * np.sin(th - phi)
                    if np.abs(d_hat) < radius + obs.radius:
                        dist_obs[idx] = np.abs(d * np.cos(th - phi))
                    else:
                        # does not intersect
                        dist_obs[idx] = self.dist_obs_max

        return dist_obs

    def cost_heading(self, v, omg, x_curr):
        """
        Calculates the heading cost given a velocity profile
        """
        # calculate positions for the next n time steps
        x, y, th = x_curr
        cost = 0.0
        cost_end = 0.0

        for k in range(self.horizon):
            x, y, th = motion_update_2d((x, y, th), (v, omg), self.dt)

            # calculate cost at the new position
            nav_dir, cost_end = self.global_planner.get_plan(x=(x, y),
                                                             interpolate=self.flag_interpolate)

            cost += np.abs(wrap_to_pi(th - self.dir_to_ang[nav_dir]))

        return cost / self.horizon / np.pi

    def cost_collision(self, dist_obs_min):
        """
        Calculates the collision cost given a velocity profile
        """
        if dist_obs_min >= self.dist_obs_max:
            return 0.0
        else:
            return 1.0 / dist_obs_min

    def cost_velocity(self, v, omg):
        """
        Calculate the velocity cost given a velocity profile
        """
        return v / self.vel_max[0]

    def admissible(self, v, omg, dist_obs_min):
        """
        Calculate whether the velocity profile is admissible
        """
        if v > np.sqrt(2.0 * dist_obs_min * self.acc_max[0]) or \
                        omg > np.sqrt(2.0 * dist_obs_min * self.acc_max[1]):
            return False
        return True

    def cost_sum(self, v, omg, x_curr):
        """
        Calculate the total cost given a velocity profile
        """
        # calculate the distance to obstacles
        dist_obs = self.update_obstacle_dists(x_curr, self.robot_radius, v, omg, self.clear_coeff * v)
        dist_min = np.min(dist_obs)

        if not self.admissible(v, omg, dist_min):
            return self.cost_max

        cost = self.w_costs['heading'] * self.cost_heading(v, omg, x_curr) + \
               self.w_costs['collision'] * self.cost_collision(dist_min) + \
               self.w_costs['velocity'] * self.cost_velocity(v, omg)
        return cost

    def solve(self, x_curr, vel_curr):
        """
        Solve for local plan
        :param x_curr: current robot state
        :param vel_curr: current robot velocity
        :return: new commanded velocity
        """
        # plan based on estimated new state after plan finishes
        x_new = motion_update_2d(x_curr, vel_curr, self.dt_plan)

        # loop through the dynamic window
        cost_min = 1e6
        cmd_vel = None
        for v in np.linspace(vel_curr[0] - self.window_size[0], vel_curr[0] + self.window_size[0]):
            for omg in np.linspace(vel_curr[1] - self.window_size[1], vel_curr[1] + self.window_size[1]):
                cost = self.cost_sum(v, omg, x_new)
                if cost < cost_min:
                    cost_min = cost
                    cmd_vel = (v, omg)

        return cmd_vel
