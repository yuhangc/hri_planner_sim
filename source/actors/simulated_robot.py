from actor_base import Actor
from source.planner.utils import *


class SimulatedDetRobot(Actor):
    def __init__(self, fp_radius=0.2, pose_init=(0.0, 0.0, 0.0), properties=None):
        Actor.__init__(self, fp_radius)

        # do not follow trajectory
        self.set_follow_trajectory(False)

        # current velocity
        self.vel_curr = np.array([0.0, 0.0])
        self.vel_goal = np.array([0.0, 0.0])

        # set pose
        self.__set_pose__(pose_init)

        # set properties
        if properties is None:
            self.acc_max = np.array([0.5, 1.5])
            self.vel_dead_zone = np.array([0.02, 0.05])
        else:
            pass

    def set_velocity(self, vel_goal_new):
        self.vel_goal = vel_goal_new

    def get_velocity(self):
        return self.vel_curr

    def update(self, dt=0.0):
        """
        Update dynamics, should be called in high frequency (> 500Hz)
        """
        self.t_curr += dt

        # update position first
        self.x, self.y, self.th = motion_update_2d((self.x, self.y, self.th),
                                                   self.vel_curr,
                                                   dt)

        # update velocity
        vel_diff = self.vel_goal - self.vel_curr
        vel_inc_max = self.acc_max * dt
        for k in range(2):
            if np.abs(vel_diff[k]) > self.vel_dead_zone[k]:
                # try to reach gaol velocity
                if vel_diff[k] < vel_inc_max[k]:
                    self.vel_curr[k] -= vel_inc_max[k]
                elif vel_diff[k] > vel_inc_max[k]:
                    self.vel_curr[k] += vel_inc_max[k]
                else:
                    self.vel_curr[k] = self.vel_goal[k]