import numpy as np

# tolerance
TOL = 1e-3


# euclidean distance
def distance(x1, x2):
    """
    Computes the euclidean distance between two states
    :param x1: first state tuple
    :param x2: second state tuple
    :return: Float euclidean distance
    """
    return np.linalg.norm(np.array(x1) - np.array(x2))


# wrap an angle to [-pi, pi)
def wrap_to_pi(ang):
    while ang < -np.pi:
        ang += 2.0 * np.pi
    while ang >= np.pi:
        ang -= 2.0 * np.pi

    return ang


# motion model of a 2D non-holonomic robot
def motion_update_2d(x_curr, vel_curr, dt):
    """
    Calculate next state based on current state and velocity
    """
    x, y, th = x_curr
    v, om = vel_curr
    th_new = th + om * dt

    if vel_curr[1] > TOL:
        x_new = x + v / om * (np.sin(th_new) - np.sin(th))
        y_new = y - v / om * (np.cos(th_new) - np.cos(th))
    else:
        x_new = x + v * np.cos(th) * dt
        y_new = y + v * np.sin(th) * dt

    return x_new, y_new, th_new


# simple obstacle class
class SimpleObstacle:
    def __init__(self, pose, vel, radius):
        self.x = pose[0:2]
        self.vel = vel
        self.radius = radius
