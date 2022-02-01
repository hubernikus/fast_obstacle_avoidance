"""
Simplified obstacle avoidance for mobile for mobile robots based on
DS & reference_direction

The algorithm is developed for shared control,
> human will know the best path, just be fast and don't get stuck
(make unexpected mistakes)
"""

import warnings

import numpy as np
from numpy import linalg as LA

from vartools.linalg import get_orthogonal_basis

from fast_obstacle_avoidance.control_robot import BaseRobot
from ._base import SingleModulationAvoider


class FastObstacleAvoider(SingleModulationAvoider):
    def __init__(self, obstacle_environment, robot: BaseRobot = None, dimension=2):
        """Initialize with obstacle list"""
        self.obstacle_environment = obstacle_environment
        self.robot = robot

        super().__init__()

        if (
            len(self.obstacle_environment) and self.obstacle_environment.dimension == 3
        ) or dimension == 2:
            from scipy.spatial.transform import Rotation as R

        self.consider_relative_velocity = True

    @property
    def dimension(self):
        return self.obstacle_environment.dimension

    def update_relative_velocity(self, weights, position):
        """Update linear and angular velocity (without deformation)."""
        linear_velocities = np.zeros((self.dimension, weights.shape[0]))
        angular_velocities = np.zeros((weights.shape[0]))

        for it, obs in enumerate(self.obstacle_environment):
            if weights[it] <= 0:
                continue

            linear_velocities[:, it] = obs.linear_velocity
            angular_velocities[it] = obs.angular_velocity

        if any(angular_velocities):
            warnings.warn("Not yet implemented for angular velocity.")

        self.relative_velocity = np.sum(
            (np.tile(weights, (linear_velocities.shape[0], 1)) * linear_velocities),
            axis=1,
        )

        return self.relative_velocity

    def update_reference_direction(self, in_robot_frame=True, position=None):
        """Take position from robot position if not given as argument."""
        if not len(self.obstacle_environment):
            # No obstacles found -> default reference
            # By default we assume dim=2
            # TODO: specified for any other case (!)
            self.reference_direction = np.zeros(2)
            self.normal_direction = np.zeros(self.reference_direction.shape)
            self.norm_angle = np.zeros(self.reference_direction.shape[0] - 1)
            self.distance_weight_sum = 0
            return

        if position is None:
            if in_robot_frame:
                position = np.zeros(self.obstacle_environment.dimension)
            else:
                position = self.robot.pose.position

        norm_dirs = np.zeros(
            (self.obstacle_environment.dimension, self.obstacle_environment.n_obstacles)
        )
        ref_dirs = np.zeros(norm_dirs.shape)
        relative_distances = np.zeros((norm_dirs.shape[1]))

        if self.consider_relative_velocity:
            relative_velocities = np.zeros(ref_dirs.shape)

        for it, obs in enumerate(self.obstacle_environment):
            norm_dirs[:, it] = obs.get_normal_direction(position, in_global_frame=True)
            ref_dirs[:, it] = (-1) * obs.get_reference_direction(
                position, in_global_frame=True
            )

            relative_distances[it] = obs.get_gamma(position, in_global_frame=True) - 1

        weights = self.get_weight_from_distances(relative_distances)

        self.reference_direction = np.sum(
            ref_dirs * np.tile(weights, (ref_dirs.shape[0], 1)), axis=1
        )

        norm_ref_dir = LA.norm(self.reference_direction)
        if not norm_ref_dir:
            self.normal_direction = np.zeros(reference_direction.shape)
            return

        self.normal_direction = self.update_normal_direction(
            ref_dirs, norm_dirs, weights
        )

        if self.consider_relative_velocity:
            self.update_relative_velocity(weights=weights, position=position)

        if self.robot is not None:
            self.robot.retrieved_obstacles()

    # def passpass():
    def update_normal_direction(self, ref_dirs, norm_dirs, weights) -> np.ndarray:
        """Update normal direction as mentioned in the paper xy."""
        delta_normals = norm_dirs - ref_dirs
        delta_normal = np.sum(
            delta_normals * np.tile(weights, (delta_normals.shape[0], 1)), axis=1
        )

        dot_prod = (-1) * (
            np.dot(delta_normal, self.reference_direction)
            / (LA.norm(delta_normal) * LA.norm(self.reference_direction))
        )

        if dot_prod < np.sqrt(2) / 2:
            normal_scaling = 1
        else:
            normal_scaling = np.sqrt(2) * dot_prod

        self.normal_direction = (
            normal_scaling
            * self.reference_direction
            / LA.norm(self.reference_direction)
            + delta_normal
        )
        self.normal_direction = self.normal_direction / LA.norm(self.normal_direction)

        return self.normal_direction

    # def update_normal_direction(self, ref_dirs, norm_dirs, weights) -> np.ndarray:
    def update_normal_direction_with_relative_rotation(
        self, ref_dirs, norm_dirs, weights
    ) -> np.ndarray:
        """Update the normal direction of an obstacle.
        This approach is based on relative rotation, it would potentially be nicer,
        but we could not extend it to d>3."""
        if self.obstacle_environment.dimension == 2:
            norm_angles = np.cross(ref_dirs, norm_dirs, axisa=0, axisb=0)

            self.norm_angle = np.sum(norm_angles * weights)
            self.norm_angle = np.arcsin(self.norm_angle)

            # Add angle to reference direction
            unit_ref_dir = self.reference_direction / LA.norm(self.reference_direction)
            self.norm_angle += np.arctan2(unit_ref_dir[1], unit_ref_dir[0])

            self.normal_direction = np.array(
                [np.cos(self.norm_angle), np.sin(self.norm_angle)]
            )

        elif self.obstacle_environment.dimension == 3:
            norm_angles = np.cross(norm_dirs, ref_dirs, axisa=0, axisb=0)

            self.norm_angle = np.sum(
                norm_angles * np.tile(weights, (relative_position.shape[0], 1)), axis=1
            )
            norm_angle_mag = LA.norm(self.norm_angle)
            if not norm_angle_mag:  # Zero value
                self.normal_direction = copy.deepcopy(self.reference_direction)

            else:
                norm_rot = Rotation.from_vec(
                    self.normal_direction / norm_angle_mag * np.arcsin(norm_angle_mag)
                )

                unit_ref_dir = self.reference_direction / norm_ref_dir

                self.normal_direction = norm_rot.apply(unit_ref_dir)

        else:
            raise NotImplementedError(
                "For higher dimensions it is currently not defined."
            )

        return self.normal_direction
