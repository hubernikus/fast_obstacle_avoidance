"""
Simple wrapper class which performs the dynamic_modulation_avoidance, but is compatible with the present false
(to not have to change the simulator-environments)
"""

from ._base import SingleModulationAvoider

from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving


class ModulationAvoider(SingleModulationAvoider):
    def __init__(self, robot, obstacle_environment, *args, **kwargs):
        super().__init__(*args, **kwargs)

        obstacle_environment = obstacle_environment

    def avoid(self, initial_velocity, position=None) -> np.ndarray:
        if position is None:
            position = self.robot.position

        modulated_velocity = obs_avoidance_interpolation_moving(
            position, initial_velocity, self.obstacle_environment
        )

        return modulated_velocity
