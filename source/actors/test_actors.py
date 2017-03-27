import unittest
import numpy as np
import matplotlib.pyplot as plt
from time import sleep

from actor_base import Actor

default_tol_place = 3


class TestActorBase(unittest.TestCase):
    def setUp(self):
        self.actor = Actor()

        traj = np.array([[0.0, 1.0, 1.0, 0.0],
                         [2.0, 3.0, -1.0, -np.pi/3.0],
                         [5.0, 0.0, -2.0, np.pi],
                         [8.0, -2.0, 1.0, np.pi/2.0],
                         [10.0, 0.0, 3.0, np.pi/3.0]])
        self.actor.load_trajectory(traj=traj)

    def assert_pose_equal(self, xd, yd, thd):
        x, y, th = self.actor.get_pose()
        self.assertAlmostEqual(x, xd, places=default_tol_place)
        self.assertAlmostEqual(y, yd, places=default_tol_place)
        self.assertAlmostEqual(th, thd, places=default_tol_place)

    def test_update0(self):
        for i in range(10):
            self.actor.update(0.1)

        self.assert_pose_equal(2.0, 0.0, -np.pi / 6.0)

    def test_update1(self):
        for i in range(35):
            self.actor.update(0.1)

        self.assert_pose_equal(1.5, -1.5, -2.0 * np.pi / 3.0)

    def test_reset0(self):
        self.actor.reset()

        self.assert_pose_equal(1.0, 1.0, 0.0)

    def test_reset1(self):
        self.actor.reset(t_reset=6.5)

        self.assert_pose_equal(-1.0, -0.5, np.pi * 0.75)

    def test_reset3(self):
        self.actor.reset(15.0)

        self.assert_pose_equal(0.0, 3.0, np.pi / 3.0)

if __name__ == "__main__":
    unittest.main()

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
    actor.plot_traj(ax)

    # animate the actor
    dt = 0.1
    for k in range(100):
        actor.plot(ax)
        plt.show()
        sleep(dt)

        actor.update(dt)
