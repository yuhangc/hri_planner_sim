import numpy as np
import matplotlib.pyplot as plt

from time import sleep

from grid_map import StochOccupancyGrid2D
from global_planners import AstarPlanner, NavFuncitonPlanner
from local_planners import DynamicWindowPlanner
from source.actors.simulated_human import SimulatedPathFollowingHuman
from source.actors.simulated_robot import SimulatedDetRobot


def test_global_planner():
    # list of obstacles
    obstacles = [((6, 6), (8, 7)), ((2, 1), (4, 2)), ((2, 4), (4, 6)), ((6, 2), (8, 4))]
    width = 10
    height = 10
    map_resolution = 0.05
    plan_resolution = 0.25

    # generate initial probabilities
    probs = -np.ones((int(width * height / map_resolution ** 2 + 0.5),))
    # probs = -np.ones((int(height / map_resolution), int(width / map_resolution)))

    # set goals
    x_init = (0, 0)
    x_goal = (8, 8)

    # create a 2D stochastic occupancy grid
    occupancy = StochOccupancyGrid2D(map_resolution, int(width / map_resolution), int(height / map_resolution),
                                     0, 0, int(plan_resolution / map_resolution) * 4, probs)
    occupancy.from_obstacles(obstacles)
    occupancy.init_map_free()

    # create a nav function planner and an astar planner
    # planner_astar = AstarPlanner((0, 0), (10, 10), x_init, x_goal, occupancy)
    planner_nav = NavFuncitonPlanner((0, 0), (10, 10), x_init, x_goal, occupancy, plan_resolution)

    # solve for path
    # planner_astar.solve(verbose=True)
    planner_nav.solve(verbose=True)
    planner_nav.reconstruct_path()

    # plot result
    # planner_astar.visualize_path()
    planner_nav.visualize_path()

    planner_nav.reconstruct_path(np.array([7, 5]))
    planner_nav.visualize_path()


def test_dwa_planner():
    # the environment
    obstacles = [((0.5, 1.5), (1.5, 3.5)), ((3.5, 1.5), (4.5, 3.5))]
    width = 5
    height = 5
    map_resolution = 0.05
    plan_resolution = 0.25

    # generate initial probabilities
    probs = -np.ones((int(width * height / map_resolution ** 2 + 0.5),))

    # create a 2D stochastic occupancy grid
    occupancy = StochOccupancyGrid2D(map_resolution, int(width / map_resolution), int(height / map_resolution),
                                     0, 0, int(plan_resolution / map_resolution) * 4, probs)
    occupancy.from_obstacles(obstacles)
    occupancy.init_map_free()

    # set goals
    x_init = (0.75, 0.75, 0.0)
    x_goal = (4.25, 4.25, 0.0)

    # construct a global planner
    planner_nav = NavFuncitonPlanner((0, 0), (width, height), x_init[0:2],
                                     x_goal[0:2], occupancy, plan_resolution)

    # get global plan
    planner_nav.solve(verbose=True)
    planner_nav.reconstruct_path()

    # visualize path
    # create a plot axis
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    plt.ion()

    planner_nav.visualize_path(ax)
    plt.pause(1)

    # create a robot
    default_robot_size = 0.2
    robot = SimulatedDetRobot(fp_radius=default_robot_size, pose_init=x_init)

    # create a local planner
    dt_local_planner = 0.1
    planner_dwa = DynamicWindowPlanner(0.2, 5,
                                       global_planner=planner_nav,
                                       robot_radius=default_robot_size)

    # starts simulation
    # simulation at 100hz
    t_curr = 0.0
    dt = 0.01
    dt_plot = 0.05

    r_local_planner = int(dt_local_planner / dt)
    r_plot = int(dt_plot / dt)

    for k in range(1500):
        # update cmd_vel
        if k % r_local_planner == 0:
            cmd_vel = planner_dwa.solve(robot.get_pose(), robot.get_velocity())
            robot.set_velocity(cmd_vel)

        # update robot state
        robot.update(dt)
        sleep(dt * 0.2)

        # plot at lower frequency
        if k % r_plot == 0:
            robot.plot(ax)
            plt.pause(0.001)

        # update simulation time
        t_curr += dt

if __name__ == "__main__":
    # test_global_planner()
    test_dwa_planner()
