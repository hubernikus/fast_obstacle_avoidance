"""
Various Robot models
"""
import os
import math

from dataclasses import dataclass, field

import numpy as np
from numpy import linalg as LA

from scipy.spatial.transform import Rotation
from scipy import ndimage

import matplotlib.pyplot as plt

from vartools.states import ObjectPose

from dynamic_obstacle_avoidance import containers
from dynamic_obstacle_avoidance.obstacles import Sphere

from .utils import laserscan_to_numpy


class BaseRobot:
    """Basic Robot Class"""

    def __init__(
        self,
        pose: ObjectPose = None,
        control_points=None,
        control_radiuses=None,
        dimension=2,
        obstacle_environment=None,
    ):
        self.dimension = dimension

        self.robot_image = None

        self._got_new_scan = False
        self._got_new_obstacles = True

        if control_points is None:
            self.control_points = np.zeros((self.dimension, 1))
        else:
            self.control_points = control_points

        if control_radiuses is None:
            self.control_radiuses = np.array([1.0])
        else:
            self.control_radiuses = control_radiuses

        if obstacle_environment is None:
            self.obstacle_environment = containers.SphereContainer()
        else:
            self.obstacle_environment = obstacle_environment

        if pose is None:
            self.pose = ObjectPose(position=np.zeros(self.dimension))
        else:
            self.pose = pose

    @property
    def rotation_matrix(self):
        cos_, sin_ = np.cos(self.pose.orientation), np.sin(self.pose.orientation)
        return np.array([[cos_, sin_], [(-1) * sin_, cos_]])

    @property
    def has_newscan(self):
        return self._got_new_scan

    @property
    def has_new_obstacles(self):
        return self._got_new_obstacles

    # Simplified environment for (only) one point
    @property
    def control_point(self):
        return self.control_points[:, 0]

    @control_point.setter
    def control_point(self, value):
        self.control_points[:, 0] = value

    @property
    def control_radius(self):
        return self.control_radiuses[0]

    @control_radius.setter
    def control_radius(self, value):
        self.control_radiuses[0] = value

    def retrieved_obstacles(self):
        self._got_new_obstacles = False

    def get_all_intensities(self):
        return np.hstack([insenties for insenties in self.intensity_data.values()])

    @property
    def num_control_points(self) -> int:
        return self.control_radiuses.shape[0]

    @property
    def control_position(self):
        return self.pose.transform_position_from_relative(self.control_point)

    def get_relative_positions_and_dists(
        self, positions: np.ndarray, in_robot_frame: bool = True
    ) -> np.ndarray:
        """Get normalized (relative) position and (relative) surface distance."""
        if in_robot_frame:
            rel_pos = positions
        else:
            rel_pos = positions - np.tile(self.pose.position, (positions.shape[1], 1)).T

        rel_dist = LA.norm(rel_pos, axis=0)

        rel_dir = rel_pos / np.tile(rel_dist, (rel_pos.shape[0], 1))
        rel_dist = rel_dist - self.control_radiuses[0]

        return rel_pos, rel_dir, rel_dist

    def plot2D(
        self,
        ax,
        num_points: int = 30,
        show_heading: bool = False,
        in_relative_frame: bool = False,
        patch_color="#add8e6",
    ) -> None:
        if self.robot_image is not None:
            raise NotImplementedError("Nothing is being done so far with robot images.")

        angles = np.linspace(0, 2 * np.pi, num_points)
        unit_circle = np.vstack((np.cos(angles), np.sin(angles)))

        for ii in range(self.control_radiuses.shape[0]):
            if in_relative_frame:
                ctrl_point = self.control_points[:, ii]
            else:
                ctrl_point = self.pose.transform_position_from_relative(
                    self.control_points[:, ii]
                )

            temp_cicle = (
                unit_circle * self.control_radiuses[ii]
                + np.tile(ctrl_point, (num_points, 1)).T
            )

            ax.plot(ctrl_point[0], ctrl_point[1], ".", color="k")

            if in_relative_frame:
                ax.plot(0, 0, "H", color="k")
            else:
                ax.plot(self.pose.position[0], self.pose.position[1], "H", color="k")

            circle = plt.Circle(
                # self.pose.position,
                ctrl_point,
                self.control_radius,
                color=patch_color,
                zorder=-1,
                alpha=0.5,
            )
            ax.add_patch(circle)

            ax.plot(temp_cicle[0, :], temp_cicle[1, :], "--", color="k", zorder=-1)

        if show_heading:
            # Create a heading triangle
            d_angle = 30 / 180.0 * math.pi
            pos_head = np.zeros((2, 3))
            pos_head[0, 0] = math.cos(d_angle) * self.control_radius
            pos_head[1, 0] = math.sin(d_angle) * self.control_radius

            pos_head[0, 1] = self.control_radius / math.cos(d_angle)

            pos_head[0, 2] = pos_head[0, 0]
            pos_head[1, 2] = (-1) * pos_head[1, 0]

            pos_head = (
                pos_head + np.tile(self.control_points[:, ii], (pos_head.shape[1], 1)).T
            )

            if not in_relative_frame:
                #     for ii in range(pos_head.shape[1]):
                #         pos_head[:, ii] = self.pose.transform_position_from_relative(
                #             pos_head[:, ii]
                #         )

                pos_head = self.pose.transform_positions_from_relative(pos_head)

            ax.plot(pos_head[0, :], pos_head[1, :], color="k", linewidth=2, zorder=-1)


# @dataclass
# class GeneralRobot(BaseRobot):
# control_points: np.ndarray
# control_radiuses: np.ndarray

# pose: ObjectPose = None
# robot_image = None


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
    def __init__(self, **kwargs):
        # self.control_radiuses: np.ndarray = np.array([0.350])
        # self.control_radiuses: np.ndarray = np.array([0.470])

        # Default values for QOLO
        super().__init__(
            control_points=np.array([[0.035, 0]]).T,
            control_radiuses=np.array([0.43]),
            dimension=2,
            **kwargs,
        )

        self.robot_image = None

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

        # self.obstacle_environment = containers.ObstacleContainer()
        # self.obstacle_environment = containers.GradientContainer()

        self.laser_data = {}
        self.robot_image = None

        self.intensity_data = {}

        # Maximum normalization - above this full repulsion is taking effect!
        self.weight_max_norm = 6.99580150e04

    def get_allscan(self, in_robot_frame=True):
        self._got_new_scan = False
        laserscan = np.hstack([scan for scan in self.laser_data.values()])

        if not in_robot_frame:
            if LA.norm(self.pose.orientation):
                laserscan = self.rotation_matrix.T @ laserscan

            if LA.norm(self.pose.position):
                # laserscan = self.pose.transform_position_from_local_to_reference(laserscan)
                laserscan = (
                    laserscan + np.tile(self.pose.position, (laserscan.shape[1], 1)).T
                )

        return laserscan

    def set_laserscan(self, data, topic_name, save_intensity=False):
        try:
            self.laser_data[topic_name] = laserscan_to_numpy(
                data, pose=self.laser_poses[topic_name]
            )
        except KeyError:
            print("Key <{topic_name}> not found; nothing was updated.")
            return

        self._got_new_scan = True

        if save_intensity:
            is_finite = np.isfinite(np.array(data.ranges))
            self.intensity_data[topic_name] = np.squeeze(np.array(data.intensities))[
                is_finite
            ]

    def set_crowdtracker(
        self,
        crowd_msg,
        sigma=7,
        reactivity=3,
        repulsion_coeff=1.5,
        human_radius=0.6,
        margin_absolut=None,
    ):
        """Update the obstacle list based on the crowd-input.

        CrowdList: List of obstacles.
        """
        # Remove all existing crowd (human) obstacles
        # Make sure not to 'overwrite' the reference (but only modify)
        if margin_absolut is None:
            margin_absolut = self.control_radiuses[0]

        it = 0
        while it < len(self.obstacle_environment):
            if (
                hasattr(self.obstacle_environment[it], "is_human")
                and self.obstacle_environment[it].is_human
            ):
                del self.obstacle_environment[it]
            else:
                it += 1

        for person in crowd_msg.tracks:
            euler = Rotation.from_quat(
                [
                    person.pose.pose.orientation.x,
                    person.pose.pose.orientation.y,
                    person.pose.pose.orientation.z,
                    person.pose.pose.orientation.w,
                ]
            ).as_euler("zyx")

            human_obs = Sphere(
                center_position=self.pose.transform_position_to_relative(
                    np.array([person.pose.pose.position.x, person.pose.pose.position.y])
                ),
                orientation=(euler[0] - self.pose.orientation),
                linear_velocity=self.pose.transform_direction_to_relative(
                    np.array([person.twist.twist.linear.x, person.twist.twist.linear.y])
                ),
                angular_velocity=0,
                tail_effect=False,
                radius=human_radius,
                margin_absolut=margin_absolut,
                # Veloctiy reduction
                reactivity=reactivity,
                repulsion_coeff=repulsion_coeff,
            )

            human_obs.is_human = True

            self.obstacle_environment.append(human_obs)  # TODO: add robot margin

        self._got_new_obstacles = True

    def plot_robot(self, ax, bag_dir="figures/qolo", length_x=1019.23 * 1e-3):
        self.length_x = length_x
        if self.robot_image is None:
            import matplotlib.image as mpimg

            self.robot_image = (
                mpimg.imread(os.path.join(bag_dir, "Qolo_T_CB_top_bumper.png")) * 255
            ).astype("uint8")

            # Length of robot
            # self.length_x = 0.92817
            # self.length_x = 1019.23 * 1e-3
            self.length_y = (
                self.robot_image.shape[0] / self.robot_image.shape[1] * self.length_x
            )

            self.pose_reference = np.array([-self.length_x / 2 * 341.23 * 1e-3, 0])

        rot = self.pose.orientation
        img_rotated = ndimage.rotate(self.robot_image, rot * 180.0 / np.pi, cval=255)

        lenght_x_rotated = (
            np.abs(np.cos(rot)) * self.length_x + np.abs(np.sin(rot)) * self.length_y
        )

        lenght_y_rotated = (
            np.abs(np.sin(rot)) * self.length_x + np.abs(np.cos(rot)) * self.length_y
        )

        pos_ref = self.rotation_matrix.T @ self.pose_reference
        pos_ref = pos_ref + self.pose.position

        ax.imshow(
            img_rotated,
            extent=[
                pos_ref[0] - lenght_x_rotated / 2.0,
                pos_ref[0] + lenght_x_rotated / 2.0,
                pos_ref[1] - lenght_y_rotated / 2.0,
                pos_ref[1] + lenght_y_rotated / 2.0,
            ],
            zorder=-2,
        )
