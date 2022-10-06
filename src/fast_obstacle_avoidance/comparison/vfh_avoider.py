""" Implementation of the vector field histogram (VFH) algorithm.

This code has been copied from:
https://github.com/vanderbiltrobotics/vfh-python

(No fork / sub-module, due to inactivity for several years and somehow very large git repo/history.)
"""

import numpy as np

# from fast_obstacle_avoidance.obstacle_avoider._base import SampledAvoider
from fast_obstacle_avoidance.obstacle_avoider.lidar_avoider import SampledAvoider

# from ._base import SingleModulationAvoider
from fast_obstacle_avoidance.comparison.vfh_python.lib.robot import Robot as VFH_Robot
from fast_obstacle_avoidance.comparison.vfh_python.lib.polar_histogram import (
    PolarHistogram,
)
from fast_obstacle_avoidance.comparison.vfh_python.lib.path_planner import (
    PathPlanner as VFH_Planner,
)


class VectorFieldHistogramAvoider(SampledAvoider):
    def __init__(
        self,
        num_angular_sectors: int = 180,
        distance_limits: float = (0.05, 2),
        robot_radius: float = 0.1,
        min_turning_radius: float = 0.1,
        safety_distance: float = 0.1,
        attractor_position: np.ndarray = None,
    ):

        self.num_angular_sectors = num_angular_sectors
        # num_angular_sectors:int = 180,
        # distance_limits: float = (0.05, 2),
        # robot_radius: float = 0.1,
        # min_turning_radius: float = 0.1,
        # safety_distance: float = 0.1,

        if attractor_position is None:
            # Assuming 2D
            attractor_position = np.zeros([0, 0])

        self.vfh_histogram = PolarHistogram(num_bins=num_angular_sectors)
        self.vfh_robot = VFH_Robot(
            target_location=attractor_position,
            init_speed=np.zeros(attractor_position.shape),
        )

    @property
    def attractor_position(self) -> np.ndarray:
        return self._attractor_position

    @attractor_position.setter
    def attractor_position(self, value) -> None:
        breakpoint()
        self.vfh_robot.set_target_discrete_location(value)
        self._attractor_position = vaue

    def update_laserscan(self):
        self.vfh_robot.histogram

    def avoid(self, position, velocity):
        self.vfh_robot.update_location(position)

        # Make a step
        self.vfh_robot.bupdate_angle()  # angle: Null (or optionally, t-1) => t
        # self.set_speed() # speed: Null (or optionally, t-1) => t

        self.vfh_robot.update_velocity()
        self.vfh_robot.update_location()  # position: t => t+1

        return self.vfh_robot.velocity


if (__name__) == "__main__":
    my_avoider = VectorFieldHistogramAvoider()

    breakpoint()
