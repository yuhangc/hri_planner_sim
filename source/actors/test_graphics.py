import numpy as np
import matplotlib.pyplot as plt

from actor_base import Actor

if __name__ == "__main__":
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
