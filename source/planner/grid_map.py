import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class StochOccupancyGrid2DBase:
    """
    2D stochastic occupancy grid, each cell stores a probability of the
    cell being occupied
    """
    def __init__(self, resolution, width, height, origin_x, origin_y,
                 window_size, probs, th_free=0.5, th_occupied=0.5):
        self.resolution = resolution
        self.width = width
        self.height = height
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.probs = probs
        self.window_size = window_size

        # threshold values to check whether a cell is free or occupied
        self.th_free = th_free
        self.th_occupied = th_occupied

    def get_grid_pos(self, x, y):
        grid_x = int(round((x - self.origin_x) / self.resolution))
        grid_y = int(round((y - self.origin_y) / self.resolution))

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

        return (1.0 - p_total) < self.th_free

    def is_free(self, state):
        """
        Check if a state is free, to be implemented in sub-classes
        """
        raise Exception("is_free method must be overriden by a subclass!")

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
        """
        plot the obstacles in red dots
        """
        pts = []
        for i in range(len(self.probs)):
            # convert i to (x,y)
            gy = int(i / self.width)
            gx = i % self.width
            x = gx * self.resolution + self.origin_x
            y = gy * self.resolution + self.origin_y
            if self.probs[i] > self.th_occupied:
                pts.append((x, y))
        pts_array = np.array(pts)
        ax.scatter(pts_array[:, 0], pts_array[:, 1], color="red", zorder=15, label='planning resolution')


class StochOccupancyGrid2D(StochOccupancyGrid2DBase):
    pass


class StochOccupancyGrid2DLazy(StochOccupancyGrid2DBase):
    def __init__(self, resolution, width, height, origin_x, origin_y,
                 window_size, probs, th_free=0.5, th_occupied=0.5):
        StochOccupancyGrid2DBase.__init__(self, resolution, width, height, origin_x, origin_y,
                                          window_size, probs, th_free, th_occupied)
        # variable used to store if the cells are free
        # 0 - free, 1 - obstacle,
        # 2 - robot base in collision with obstacle
        # -1 - unknown
        self.free = -np.ones((self.width, self.height), dtype=int)

    def is_free(self, state):
        """
        Lazy check if a state is free, direct return the pre-computed value
        """
        grid_x, grid_y = self.get_grid_pos(state[0], state[1])
        return self.free[grid_x, grid_y] == 0

    def is_boundary(self, x, y):
        """
        Check if a cell is on the boundary of an obstacle with 4-connected model
        """
        if (self.free[x - 1, y] != 1) or (self.free[x, y - 1] != 1) or \
                (self.free[x + 1, y] != 1) or (self.free[x, y + 1] != 1):
            return True
        else:
            return False

    def inflate_obstacle(self, x, y, radius):
        """
        Inflate the occupied grid by radius
        """
        for yy in range(int(np.floor(y - radius)), int(np.ceil(y + radius))):
            if 0 <= yy < self.height:
                dxx = np.sqrt(radius**2 - (yy - y)**2)
                for xx in range(int(np.floor(x - dxx)), int(np.ceil(x + dxx))):
                    if 0 <= xx < self.width:
                        if self.free[xx, yy] == 0:
                            self.free[xx, yy] = 2

    def compute_free_states(self, exp_radius=None):
        """
        Pre-compute if the state is free for all states in the map
        Result stored in self.free
        """
        for x in range(self.width):
            for y in range(self.height):
                if self._is_free((x, y), physical_state=False):
                    self.free[x, y] = 0
                else:
                    self.free[x, y] = 1

        # extend obstacles by certain radius
        if exp_radius is not None:
            grid_radius = int(exp_radius / self.resolution)
            for x in range(1, self.width - 1):
                for y in range(1, self.height - 1):
                    if self.free[x, y] == 1 and self.is_boundary(x, y):
                        self.inflate_obstacle(x, y, grid_radius)

    def plot(self, ax):
        """
        Plot obstacle and extended area differently
        """
        pts_obs = []
        pts_ext = []

        for gx in range(self.width):
            for gy in range(self.height):
                x = gx * self.resolution + self.origin_x
                y = gy * self.resolution + self.origin_y

                if self.free[gx, gy] == 1:
                    pts_obs.append((x, y))
                elif self.free[gx, gy] == 2:
                    pts_ext.append((x, y))

        pts_array = np.array(pts_obs)
        ax.scatter(pts_array[:, 0], pts_array[:, 1], color="red", zorder=2, label='obstacle area')

        pts_array = np.array(pts_ext)
        if len(pts_array) > 0:
            ax.scatter(pts_array[:, 0], pts_array[:, 1], color="orange", zorder=2, label='extended obstacle area')
