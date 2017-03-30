import numpy as np

from actor_base import Actor


class SimulatedPathFollowingHuman(Actor):
    """
    A simulated human object that only follows a given path
    """
    def __init__(self):
        Actor.__init__(self)


class SimulatedReactiveHuman(Actor):
    """
    A reactive human object that follows a path but reacts
    to robot's movement
    """
    def __init__(self):
        Actor.__init__(self)
        self.react_radius_large = 0.5
        self.react_radius_small = 0.2

    def update(self, dt=0.0):
        pass
