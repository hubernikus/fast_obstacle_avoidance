"""
Simple wrapper class which performs the dynamic_modulation_avoidance, but is compatible with the present false
(to not have to change the simulator-environments)
"""

import warnings
import numpy as np

from ._base import SingleModulationAvoider

from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving


class ModulationAvoider(SingleModulationAvoider):
    def __init__(self, robot, obstacle_environment, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.robot = robot
        self.obstacle_environment = obstacle_environment

        self.dimension = 2

    @property
    def reference_direction(self):
        return np.zeros(self.dimension)

    @reference_direction.setter
    def reference_direction(self, value):
        warnings.warn("Dummy setter - nothing is executed.")

    def avoid(self, initial_velocity, position=None):
        if position is None:
            position = self.robot.pose.position

        modulated_velocity = obs_avoidance_interpolation_moving(
            position, initial_velocity, self.obstacle_environment
        )

        return modulated_velocity

    def update_reference_direction(self, *args, **kwargs):
        warnings.warn("Empty (compatibility) method - everything is done in the avoid.")

    def update_laserscan(self, *args, **kwargs):
        warnings.warn("Empty (compatibility) method - everything is done in the avoid.")

    # def update_reference_direction(self, *args, **kwargs):
    #     warnings.warn("Empty (compatibility) method - everything is done in the avoid.")
