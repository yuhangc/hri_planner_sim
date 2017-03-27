import numpy as np
import matplotlib.patches as patches


def interp_path_2d(path, counter, t):
    """
    Interpolating between two 2D path points
    """
    result = np.zeros((3, ))
    xp = np.array([path[counter-1, 0], path[counter, 0]])

    result[0] = np.interp(t, xp, np.array([path[counter - 1, 1], path[counter, 1]]))
    result[1] = np.interp(t, xp, np.array([path[counter - 1, 2], path[counter, 2]]))

    # interpolate the rotation in 2d
    result[2] = interp_so2(t, xp[0], xp[1], path[counter-1, 3], path[counter, 3])

    return result


def interp_so2(t, t0, t1, rot0, rot1):
    """
    Interpolating between two 2D rotations
    """
    drot = rot1 - rot0
    # wrap to pi
    drot = (drot + np.pi) % (2.0 * np.pi) - np.pi
    rot = rot0 + (t - t0) / (t1 - t0) * drot
    # wrap to pi
    rot = (rot + np.pi) % (2.0 * np.pi) - np.pi

    return rot


class Actor:
    """
    A base class, defines an actor that can follow a given path in 2D
    Assuming a holonomic actor, i.e. can move in all directions
    """
    def __init__(self, fp_radius=0.3):
        self.x = 0.0
        self.y = 0.0
        self.th = 0.0

        self.path = None
        self.flag_follow_traj = True

        self.t_curr = 0.0
        self.counter = 0
        self.len_traj = 0

        self.fp_radius = fp_radius
        self.footprint = None
        self.fp_orientation = None

    def __set_pose__(self, new_pose):
        self.x, self.y, self.th = new_pose

    def load_trajectory(self, traj=None, file_path=None):
        """
        Set the trajectory for following
        :param traj: numpy array for the new trajectory
        :param file_path: file path for the trajectory data file
        """
        if traj is None:
            # load from file
            pass
        else:
            # from a numpy array
            self.path = traj
            self.len_traj = traj.shape[0]

            # automatically set the pose to the first way point
            self.__set_pose__(traj[0][1:])

    def set_follow_trajectory(self, flag_follow_traj):
        """
        Set the flag for follow trajectory
        :param flag_follow_traj: boolean variable for new flag
        """
        self.flag_follow_traj = flag_follow_traj

    def reset(self, t_reset=0.0):
        """
        Reset the actor to a certain time
        :param t_reset: time to reset to
        """
        if t_reset < 0.0:
            # raise some error
            raise Exception("Can't set to time < 0!")

        self.counter = 0
        self.t_curr = t_reset

        self.update()

    def update(self, dt=0.0):
        """
        For simple path follow actor, here just update time
        :param dt: time increment
        """
        self.t_curr += dt

        if self.path is None:
            raise Exception("No path set yet!")

        while self.counter < self.len_traj and self.path[self.counter, 0] <= self.t_curr:
            self.counter += 1
        if self.counter < self.len_traj:
            # interpolate
            self.__set_pose__(interp_path_2d(self.path, self.counter, self.t_curr))
        else:
            # set pose to last point
            self.__set_pose__(self.path[-1, 1:])

    def get_pose(self):
        return self.x, self.y, self.th

    def plot(self, ax):
        """
        Plot the graphics representation of the human, a circle with orientation
        :param ax: the axis to plot on
        """
        if self.footprint is None:
            # create new plot
            self.footprint = ax.add_patch(
                patches.Circle(
                    (self.x, self.y), self.fp_radius,
                    facecolor="red", alpha=0.5
                )
            )
            # draw the orientation
            x_plot = np.array([self.x, self.x + self.fp_radius * np.cos(self.th)])
            y_plot = np.array([self.y, self.y + self.fp_radius * np.sin(self.th)])
            self.fp_orientation = ax.plot(x_plot, y_plot, lw=2, ls='-')
        else:
            # update the current drawing
            self.footprint.center = (self.x, self.y)
            x_plot = np.array([self.x, self.x + self.fp_radius * np.cos(self.th)])
            y_plot = np.array([self.y, self.y + self.fp_radius * np.sin(self.th)])
            self.fp_orientation[0].set_xdata(x_plot)
            self.fp_orientation[0].set_ydata(y_plot)

    def plot_traj(self, ax):
        """
        Plot the trajectory for following
        :param ax: the axis to plot on
        """
        if self.path is None:
            raise Exception("No path set yet!")

        ax.plot(self.path[:, 1], self.path[:, 2], 'b-', lw=2)
