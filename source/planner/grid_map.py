import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# A 2D state space grid with a set of rectangular obstacles. The grid is fully deterministic
class DetOccupancyGrid2D(object):
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = obstacles

    def is_free(self, x):
        for obs in self.obstacles:
            inside = True
            for dim in range(len(x)):
                if x[dim] < obs[0][dim] or x[dim] > obs[1][dim]:
                    inside = False
                    break
            if inside:
                return False
        return True

    def plot(self, fig_num=0):
        fig = plt.figure(fig_num)
        for obs in self.obstacles:
            ax = fig.add_subplot(111, aspect='equal')
            ax.add_patch(
                patches.Rectangle(
                    obs[0],
                    obs[1][0] - obs[0][0],
                    obs[1][1] - obs[0][1], ))


class StochOccupancyGrid2D(object):
    """
    2D stochastic occupancy grid, each cell stores a probability of the
    cell being occupied
    """
    def __init__(self, resolution, width, height, origin_x, origin_y,
                 window_size, probs, thresh=0.5):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.probs = probs
        self.window_size = window_size
        self.thresh = thresh

        # variable used to store if states are free
        # zero is not free
        self.free = np.zeros((self.width, self.height), dtype=int)

    def get_grid_pos(self, x, y):
        grid_x = int((x - self.origin_x) / self.resolution)
        grid_y = int((y - self.origin_y) / self.resolution)

        return grid_x, grid_y

    def _is_free(self, state, physical_state=True, window_size=None):
        """
        Check if a state is free by looking at a certain window around the state
        Combine the probabilities of each cell by assuming independence of each
        estimation
        """
        # TODO: to be optimized
        # use default window size if not specified
        if window_size is None:
            window_size = self.window_size

        p_total = 1.0
        lower = -int(round((window_size - 1) / 2))
        upper = int(round((window_size - 1) / 2))
        for dx in range(lower, upper + 1):
            for dy in range(lower, upper + 1):
                if physical_state:
                    # convert to grid state
                    x = state[0] + dx * self.resolution
                    y = state[1] + dy * self.resolution
                    grid_x, grid_y = self.get_grid_pos(x, y)
                else:
                    grid_x = state[0] + dx
                    grid_y = state[1] + dy
                if 0 < grid_x < self.width and 0 < grid_y < self.height:
                    p_total *= (1.0 - max(0.0, float(self.probs[grid_y * self.width + grid_x]) / 100.0))

        return (1.0 - p_total) < self.thresh

    def is_free(self, state):
        """
        Check if a state is free, directly return the pre-computed self.free
        """
        grid_x, grid_y = self.get_grid_pos(state[0], state[1])
        return self.free[grid_x, grid_y] > 0

    def init_map_free(self):
        """
        Pre-compute if the state is free for all states in the map
        Result stored in self.free
        """
        for x in range(self.width):
            for y in range(self.height):
                if self._is_free((x, y), physical_state=False):
                    self.free[x, y] = 1
                else:
                    self.free[x, y] = 0

    def from_obstacles(self, obstacles):
        """
        Construct stochastic occupancy grid from list of rectangular obstacles
        :param obstacles: list of rectangular obstacles
        """
        # clear the occupancy grid
        self.probs = np.zeros_like(self.probs)

        for obs in obstacles:
            x_start, y_start = self.get_grid_pos(obs[0][0], obs[0][1])
            x_end, y_end = self.get_grid_pos(obs[1][0], obs[1][1])
            for grid_x in range(x_start, x_end+1):
                for grid_y in range(y_start, y_end+1):
                    self.probs[grid_y * self.width + grid_x] = 100

    def plot(self, ax):
        pts = []
        for i in range(len(self.probs)):
            # convert i to (x,y)
            gy = int(i / self.width)
            gx = i % self.width
            x = gx * self.resolution + self.origin_x
            y = gy * self.resolution + self.origin_y
            # if not self.is_free((x, y)):
            #     pts.append((x, y))
            if self.probs[i] > self.thresh:
                pts.append((x, y))
        pts_array = np.array(pts)
        ax.scatter(pts_array[:, 0], pts_array[:, 1], color="red", zorder=15, label='planning resolution')
