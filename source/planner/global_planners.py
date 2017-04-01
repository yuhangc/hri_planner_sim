import numpy as np
import matplotlib.pyplot as plt

from collections import deque
from utils import distance


class GlobalPlannerBase:
    def __init__(self, statespace_lo, statespace_hi, x_init=None, x_goal=None, occupancy=None, resolution=1.0):
        """
        Initialize the planner
        :param statespace_lo: physical state space lower bound
        :param statespace_hi: physical state space upper bound
        :param x_init: physical initial state
        :param x_goal: physical initial state
        :param occupancy: the occupancy grid map
        :param resolution: resolution of the discretization of planning state space
        """
        self.occupancy = occupancy
        self.resolution = resolution
        # convert everything to planning grid space
        self.statespace_lo = self.snap_to_grid(statespace_lo)
        self.statespace_hi = self.snap_to_grid(statespace_hi)
        self.x_init = self.snap_to_grid(x_init)
        self.x_goal = self.snap_to_grid(x_goal)

        # stores the global path to goal
        self.path = None

        # flag that indicates whether there is a plan
        self.has_plan = False

    def _set_x_init(self, x_init):
        # raise error if not free
        if not self.is_free(x_init):
            raise Exception("Initial state has to be in free space!")
        self.x_init = self.snap_to_grid(x_init)

    def _set_x_goal(self, x_goal):
        # raise error if not free
        if not self.is_free(x_goal):
            raise Exception("Goal state has to be in free space!")
        self.x_goal = self.snap_to_grid(x_goal)

    def _update_occupancy(self, occupancy):
        self.occupancy = occupancy

    def reset(self, x_init=None, x_goal=None, occupancy=None):
        """
        Optionally reset the initial state, goal state and occupancy map
        """
        if x_init is not None:
            self._set_x_init(x_init)
        if x_goal is not None:
            self._set_x_goal(x_goal)
        if occupancy is not None:
            self._update_occupancy()

    def is_free(self, x):
        """
        Checks if a give state is free, meaning it is inside the bounds of the map and
        is not inside any obstacle
        :param x: tuple/np array state
        :return: True/False
        """
        x = np.array(x)
        if np.all(x == self.x_init) or np.all(x == self.x_goal):
            return True

        for dim in range(len(x)):
            if x[dim] < self.statespace_lo[dim]:
                return False
            if x[dim] >= self.statespace_hi[dim]:
                return False

        # occupancy map takes in physical state instead of grid state
        if not self.occupancy.is_free(x * self.resolution):
            return False

        return True

    def snap_to_grid(self, x):
        """
        Snap state x to map grid
        :param x: 1D array-like input (tuple or np array)
        """
        return np.rint(np.array(x) / self.resolution).astype(int)

    def solve(self, max_iter=10000, verbose=False):
        """
        Solve the global planning problem, must be implemented by subclasses
        :param max_iter: maximum iteration for solving the problem
        :param verbose: print intermediate information when solving
        """
        raise NotImplementedError("solve method must be overriden by a subclass!")

    def get_path(self):
        """
        returns the planned path, directly return self.path
        """
        return self.path

    def visualize_path(self, ax):
        """
        Plots the path found in self.path and the obstacles
        """
        if not self.path:
            return

        self.occupancy.plot(ax)

        solution_path = np.array(self.path) * self.resolution
        ax.plot(solution_path[:, 0], solution_path[:, 1],
                 color="green", linewidth=2, label="solution path", zorder=10)
        ax.scatter([self.x_init[0] * self.resolution, self.x_goal[0] * self.resolution],
                    [self.x_init[1] * self.resolution, self.x_goal[1] * self.resolution],
                    color="green", s=30, zorder=10)
        ax.annotate(r"$x_{init}$", np.array(self.x_init) * self.resolution + np.array([.2, 0]), fontsize=16)
        ax.annotate(r"$x_{goal}$", np.array(self.x_goal) * self.resolution + np.array([.2, 0]), fontsize=16)
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.03), fancybox=True, ncol=3)


class AstarPlanner(GlobalPlannerBase):
    """
    A* path planning, finds global shortest path to goal
    """
    def __init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution=1):
        GlobalPlannerBase.__init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution)

        self.closed_set = []  # the set containing the states that have been visited
        self.open_set = []  # the set containing the states that are condidate for future expension

        self.f_score = {}  # dictionary of the f score (estimated cost from start to goal passing through state)
        self.g_score = {}  # dictionary of the g score (cost-to-go from start to state)
        self.came_from = {}  # dictionary keeping track of each state's parent to reconstruct the path

        self.open_set.append(x_init)
        self.g_score[x_init] = 0
        self.f_score[x_init] = distance(x_init, x_goal)

        # connected neighbors
        self.dx = np.array([[1, 0], [0, 1], [0, -1], [-1, 0],
                            [1, 1], [1, -1], [-1, 1], [-1, -1]])

    def get_neighbors(self, x):
        """
        gets the FREE neighbor states of a given state. Assumes a motion model
        where we can move up, down, left, right, or along the diagonals
        by an amount equal to self.resolution (this is important for problem 3!). Uses
        the function self.is_free in order to check if any given state is indeed free.
        :param x: tuple state
        :return: List of neighbors that are free, as a list of TUPLES
        """

        free_neighbors = []
        for k in range(len(self.dx)):
            x_new = x + self.dx[k]
            if self.is_free(x_new):
                free_neighbors.append(x_new)

        return free_neighbors

    def find_best_f_score(self):
        """
        Gets the state in open_set that has the lowest f_score
        :return: A tuple, the state found in open_set that has the lowest f_score
        """
        return min(self.open_set, key=lambda x: self.f_score[x])

    def reconstruct_path(self):
        """
        Use the came_from map to reconstruct a path from the initial location
        to the goal location
        :return: A list of tuples, which is a list of the states that go from start to goal
        """
        path = [self.x_goal]
        current = path[-1]
        while current != self.x_init:
            path.append(self.came_from[current])
            current = path[-1]
        return list(reversed(path))

    def solve(self, max_iter=10000, verbose=False):
        """
        Solves the planning problem using the A* search algorithm. It places
        the solution as a list of of tuples (each representing a state) that go
        from self.x_init to self.x_goal inside the variable self.path
        :return: Boolean, True if a solution from x_init to x_goal was found
        """
        counter = 0
        while len(self.open_set) > 0:
            x_curr = self.find_best_f_score()
            if np.all(x_curr == self.x_goal):
                self.path = self.reconstruct_path()
                if verbose:
                    print "found path!"
                return True

            self.open_set.remove(x_curr)
            self.closed_set.append(x_curr)

            counter += 1
            if counter % 10 == 0 and verbose:
                # print some information for debugging
                print counter, x_curr, self.x_goal

            if counter > max_iter:
                if verbose:
                    print "maximum iteration reached, no path found!"
                return False

            for x_next in self.get_neighbors(x_curr):
                aout = self.closed_set.count(x_next)
                if self.closed_set.count(x_next):
                    continue

                g_score_new = self.g_score[x_curr] + distance(x_curr, x_next)
                if not self.open_set.count(x_next):
                    self.open_set.append(x_next)
                elif self.g_score[x_next] < g_score_new:
                    continue

                self.came_from[x_next] = x_curr
                self.g_score[x_next] = g_score_new
                self.f_score[x_next] = g_score_new + distance(x_next, self.x_goal)

        return False


class NavFuncitonPlanner(GlobalPlannerBase):
    """
    A global planner that calculates the navigation function given a goal state.
    """
    def __init__(self, statespace_lo, statespace_hi, x_init=None, x_goal=None, occupancy=None, resolution=1.0):
        GlobalPlannerBase.__init__(self, statespace_lo, statespace_hi, x_init, x_goal, occupancy, resolution)

        # calculate the shape of the planning state space
        self.statespace_shape = self.statespace_hi - self.statespace_lo
        self.dim = len(self.statespace_shape)

        # nav function stores the next state to go for each state
        self.nav_func = np.zeros(self.statespace_shape, dtype=int)
        self.shortest = -np.ones(self.statespace_shape)

        # connected neighbors
        self.dx = np.array([[1, 0], [0, 1], [0, -1], [-1, 0],
                            [1, 1], [1, -1], [-1, 1], [-1, -1]])

    def solve(self, max_iter=None, verbose=False):
        """
        Use BFS to find shortest paths from all points to goal position
        """
        # raise error if goal or map not specified
        if (self.x_goal is None) or (self.occupancy is None):
            print "goal state or occupancy map not defined!"
            return False

        # use a queue to store the expanded nodes/states
        node_list = deque()
        node_list.append(self.x_goal)
        self.shortest[tuple(self.x_goal)] = 0.0

        while node_list:
            x_curr = node_list.popleft()
            # find all neighbors of x_curr
            for k in range(len(self.dx)):
                x_new = tuple(x_curr + self.dx[k])
                if self.is_free(x_new) and self.shortest[x_new] < 0:
                    # expand to x_new if it is free and not visited before
                    self.shortest[x_new] = self.shortest[tuple(x_curr)] + distance(x_curr, x_new)
                    self.nav_func[x_new] = k
                    node_list.append(np.array(x_new))

        self.has_plan = True

        return True

    def reconstruct_path(self, x_init=None):
        """
        (Re)construct a path from a given initial state to the goal state
        :param x_init: physical (x, y) of the initial state
        :return: the constructed path self.path
        """
        if x_init is not None:
            self.x_init = self.snap_to_grid(x_init)

        # raise error if not free
        if not self.is_free(self.x_init):
            raise Exception("Starting state is not in free space!")

        self.path = [self.x_init]
        x_curr = self.x_init

        while np.any(x_curr != self.x_goal):
            dx = self.dx[self.nav_func[tuple(x_curr)]]
            x_curr = self.path[-1] - dx
            self.path.append(x_curr)

        return self.path

    def get_plan(self, x=None, interpolate=False):
        """
        Returns the navigation function and the shortest dist to goal
        :param x: state, optional input
        :param interpolate: whether to interpolate between states
        :return: the whole plan or plan of a specific state
        """
        if x is None:
            return self.nav_func, self.shortest
        else:
            if interpolate:
                # TODO: to be implemented
                pass
            else:
                grid_x, grid_y = self.snap_to_grid(x)
                return self.nav_func[grid_x, grid_y], self.shortest[grid_x, grid_y]
