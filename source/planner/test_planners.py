import numpy as np

from grid_map import StochOccupancyGrid2D
from global_planners import AstarPlanner, NavFuncitonPlanner

if __name__ == "__main__":
    # list of obstacles
    obstacles = [((6, 6), (8, 7)), ((2, 1), (4, 2)), ((2, 4), (4, 6)), ((6, 2), (8, 4))]
    width = 10
    height = 10
    map_resolution = 0.05
    plan_resolution = 0.25

    # generate initial probabilities
    probs = -np.ones((int(width * height / map_resolution**2 + 0.5), ))
    # probs = -np.ones((int(height / map_resolution), int(width / map_resolution)))

    # set goals
    x_init = (0, 0)
    x_goal = (8, 8)

    # create a 2D stochastic occupancy grid
    occupancy = StochOccupancyGrid2D(map_resolution, int(width / map_resolution), int(height / map_resolution),
                                     0, 0, int(plan_resolution/map_resolution) * 4, probs)
    occupancy.from_obstacles(obstacles)

    # create a nav function planner and an astar planner
    # planner_astar = AstarPlanner((0, 0), (10, 10), x_init, x_goal, occupancy)
    planner_nav = NavFuncitonPlanner((0, 0), (10, 10), x_init, x_goal, occupancy, plan_resolution)

    # solve for path
    # planner_astar.solve(verbose=True)
    planner_nav.solve(verbose=True)
    planner_nav.construct_path()

    # plot result
    # planner_astar.visualize_path()
    planner_nav.visualize_path()

    planner_nav.construct_path(np.array([7, 5]))
    planner_nav.visualize_path()
