"""
Various Robot models
"""
from dataclasses import dataclass, field

import numpy as np
from numpy import linalg as LA

from vartools.states import ObjectPose

from .utils import laserscan_to_numpy


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

            ax.plot(ctrl_point[0], ctrl_point[1], ".", color="k")

        ax.plot(self.pose.position[0], self.pose.position[1], "H", color="k")


@dataclass
class GeneralRobot(BaseRobot):
    control_points: np.ndarray
    control_radiuses: np.ndarray

    pose: ObjectPose = None
    robot_image = None


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

    # [WARNING] (min) lidar radius is around 0.47 - BUT door width is 0.97
    def __init__(self, pose: ObjectPose = None):
        self._got_new_scan = False
        self.pose = pose

        # self.control_points: np.ndarray = np.array([[0.035, 0]]).T
        self.control_points: np.ndarray = np.array([[0.035, 0]]).T
        # self.control_radiuses: np.ndarray = np.array([0.350])
        # self.control_radiuses: np.ndarray = np.array([0.470])
        self.control_radiuses: np.ndarray = np.array([0.43])

        self.laser_poses: dict = {
            "/front_lidar/scan": ObjectPose(
                position=np.array([0.035, 0]), orientation=0
            ),
            "/rear_lidar/scan": ObjectPose(
                position=np.array([-0.505, 0]), orientation=np.pi
            ),
            # '/front_lidar/scan': ObjectPose(position=np.array([0, 0]), orientation=0),
            # '/rear_lidar/scan': ObjectPose(position=np.array([0, 0]), orientation=np.pi),
        }

        self.laser_data = {}
        self.robot_image = None

    @property
    def has_newscan(self):
        return self._got_new_scan

    def get_allscan(self):
        self._got_new_scan = False
        return np.hstack([scan for scan in self.laser_data.values()])

    def set_laserscan(self, data, topic_name):
        try:
            self.laser_data[topic_name] = laserscan_to_numpy(
                data, pose=self.laser_poses[topic_name]
            )
            self._got_new_scan = True

        except KeyError:
            print("Key <{topic_name}> not found; nothing was updated.")
            return
