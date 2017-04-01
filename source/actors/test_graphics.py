import numpy as np
import matplotlib.pyplot as plt

from time import sleep
from actor_base import Actor
from simulated_robot import SimulatedDetRobot


def test_actor_plotting():
    # test the plotting
    actor = Actor()
    traj = np.array([[0.0, 1.0, 1.0, 0.0],
                     [2.0, 3.0, -1.0, -np.pi / 3.0],
                     [5.0, 0.0, -2.0, np.pi],
                     [8.0, -2.0, 1.0, np.pi / 2.0],
                     [10.0, 0.0, 3.0, np.pi / 3.0]])
    actor.load_trajectory(traj=traj)

    # create a plot axis
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    plt.ion()
    actor.plot_traj(ax)

    # animate the actor
    dt = 0.1
    for k in range(100):
        actor.plot(ax)
        # plt.show(block=False)
        plt.pause(0.05)

        actor.update(dt)


def test_simulated_det_robot():
    robot = SimulatedDetRobot()

    # create a plot axis
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    plt.ion()

    # motion list
    cmd_vel_list = np.array([[0.0, 0.0, 0.0],
                             [0.5, 0.5, 0.0],
                             [2.5, 0.3, 0.5],
                             [5.0, 0.0, -1.0],
                             [7.0, 0.0, 0.0]])

    # simulation at 100hz
    t_curr = 0.0
    dt = 0.01
    vel_idx = 0
    for k in range(800):
        # update cmd_vel
        if vel_idx < len(cmd_vel_list) and t_curr >= cmd_vel_list[vel_idx, 0]:
            robot.set_velocity(cmd_vel_list[vel_idx, 1:])
            vel_idx += 1

        # update robot state
        robot.update(dt)
        sleep(dt * 0.2)

        # plot at lower frequency
        if k % 5 == 0:
            robot.plot(ax)
            plt.pause(0.001)

        # update simulation time
        t_curr += dt

if __name__ == "__main__":
    # test_actor_plotting()
    test_simulated_det_robot()