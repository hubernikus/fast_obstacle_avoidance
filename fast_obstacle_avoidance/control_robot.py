"""
Various Robot models
"""
from dataclasses import dataclass

import numpy as np
from numpy import linalg as LA

from vartools.states import ObjectPose


class BaseRobot:
    """Basic Robot Class"""

    @property
    def num_control_points(self) -> int:
        return self.control_radiuses.shape[0]

    def get_relative_positions_and_dists(self, positions: np.ndarray) -> np.ndarray:
        """Get normalized (relative) position and (relative) surface distance."""
        rel_pos = positions - np.tile(self.pose.position, (positions.shape[1], 1)).T
        rel_dist = LA.norm(rel_pos, axis=0)

        rel_dir = rel_pos / np.tile(rel_dist, (rel_pos.shape[0], 1))
        rel_dist = rel_dist - self.control_radiuses[0]

        return rel_pos, rel_dir, rel_dist

    def plot2D(self, ax, num_points: int = 30) -> None:
        if self.robot_image is not None:
            raise NotImplementedError("Nothing is being done so far for robot images.")

        angles = np.linspace(0, 2 * np.pi, num_points)
        unit_circle = np.vstack((np.cos(angles), np.sin(angles)))

        for ii in range(self.control_radiuses.shape[0]):
            ctrl_point = self.pose.transform_position_from_local_to_reference(
                self.control_points[:, ii]
            )
            temp_cicle = (
                unit_circle * self.control_radiuses[ii]
                + np.tile(ctrl_point, (num_points, 1)).T
            )

            ax.plot(temp_cicle[0, :], temp_cicle[1, :], "--", color="k")


@dataclass
class GeneralRobot(BaseRobot):
    control_points: np.ndarray
    control_radiuses: np.ndarray

    pose: ObjectPose = None
    robot_image = None


@dataclass
class QoloRobot(BaseRobot):
    """
    The Qolo has following properties [m]

    Center: above wheel-axis

    Radius Bumper: 0.34123
    Lidar front: [0.035, 0]
    Lidar back: [0.505, 0]

    Wheel position (front): [0, +/- 0.545/2]
    Wheel distance (front): [-0.605, +/- 0.200/2]
    """

    pose: ObjectPose = None

    control_points: np.ndarray = np.array([[0.035, 0]])
    control_radiuses: np.ndarray = np.array([350])

    laser_positions: dict = {
        "/front_lidar/scan": np.array([0.035, 0]),
        "/rear_lidar/scan": np.array([-0.505, 0]),
    }

    robot_image = None
