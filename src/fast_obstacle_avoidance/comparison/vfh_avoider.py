""" Implementation of the vector field histogram (VFH) algorithm.

This code has been copied from:
https://github.com/vanderbiltrobotics/vfh-python

(No fork / sub-module, due to inactivity for several years and somehow very large git repo/history.)
"""

# from ._base import SingleModulationAvoider
from .vfh_python.lib.robot import VFH_Robot
from .vfh_python.lib.path_planner import PathPlanner as VFH_Planner


class VectorFieldHistogramAvoider(SampledAvoider):
    def __init__(
        self,
        num_angular_sectors: int = 180,
        distance_limits: float = (0.05, 2),
        robot_radius: float = 0.1,
        min_turning_radius: float = 0.1,
        safety_distance: float = 0.1,
    ):

        self.num_angular_sectors = num_angular_sectors
        # num_angular_sectors:int = 180,
        # distance_limits: float = (0.05, 2),
        # robot_radius: float = 0.1,
        # min_turning_radius: float = 0.1,
        # safety_distance: float = 0.1,

        self.vfh_histogram = PolarHistogram(num_bins=num_angular_sectors)
        self.vfh_robot = VFH_Robot()

    def update_laserscan(self):
        self.vfh_robot.histogram

    def avoid(self, position, velocity):
        self.vfh_robot.init_position = position

        # Make a step
        self.vfh_robot.bupdate_angle()  # angle: Null (or optionally, t-1) => t
        # self.set_speed() # speed: Null (or optionally, t-1) => t

        self.vfh_robot.update_velocity()
        self.vfh_robot.update_location()  # position: t => t+1

        return self.vfh_robot.velocity
