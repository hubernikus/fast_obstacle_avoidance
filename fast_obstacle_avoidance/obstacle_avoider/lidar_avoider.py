"""
Obstacle Avoider Dedicated to Lidar Data
"""

from __future__ import annotations

import warnings
from typing import Optional

from timeit import default_timer as timer

import math

import numpy as np
from numpy import linalg as LA

from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum

from fast_obstacle_avoidance.control_robot import BaseRobot

from ._base import SingleModulationAvoider
from .stretching_matrix import StretchingMatrixTrigonometric


def get_relative_positions_and_dists(
    center_position: np.ndarray,
    control_radius: float,
    datapoints: np.ndarray,
    in_local_frame: bool = True,
) -> np.ndarray:
    """Get normalized (relative) position and (relative) surface distance."""
    if in_local_frame:
        rel_pos = datapoints
    else:
        rel_pos = datapoints - np.tile(center_position, (datapoints.shape[1], 1)).T

    rel_dist = LA.norm(rel_pos, axis=0)

    rel_dir = rel_pos / np.tile(rel_dist, (rel_pos.shape[0], 1))
    rel_dist = rel_dist - control_radius

    return rel_pos, rel_dir, rel_dist


class SampledAvoider(SingleModulationAvoider):
    """
    To proof:
    -> reference direction becomes normal (!) when getting very close (!)
    -> obstacle is being avoided when very close(!) [reference -> normal]
    -> No local minima (and maybe even convergence to attractor)
    -> add slight repulsion along the normal direction (when getting closer)

    TODO:
    > put into 'global' frame, robot updates faster than laserscan;
    >>> hence global frame is needed
    """

    def __init__(
        self,
        robot: BaseRobot,
        evaluate_normal: bool = False,
        # delta_sampling: float = delta_sampling
        *args,
        **kwargs,
    ) -> None:
        self.robot = robot

        self.evaluate_normal = evaluate_normal
        self.max_angle_ref_norm = 80 * np.pi / 180

        # For the moment, delta_sampling is not used
        # self.delta_sampling = delta_sampling
        super().__init__(
            *args,
            **kwargs,
        )

        if robot is None:
            self._laserscan_in_robot_frame = False
        else:
            self._laserscan_in_robot_frame = True

    @property
    def datapoints(self):
        # Property to make consistent with mixed avoider.
        # But change this variable name in the future
        return self.laser_scan

    @datapoints.setter
    def datapoints(self, value):
        self.laser_scan = value

    @property
    def laserscan(self):
        # Property to make consistent with mixed avoider.
        return self.laser_scan

    @laserscan.setter
    def laserscan(self, value):
        self.laser_scan = value

    def update_laserscan(self, laserscan=None, in_robot_frame=None):
        if in_robot_frame is not None:
            self._laserscan_in_robot_frame = in_robot_frame
        # if in_robot_frame is False:
        # raise NotImplementedError()

        if laserscan is not None:
            self.laserscan = laserscan
            self._got_new_scan = True

        elif self.robot.has_newscan:
            self.laserscan = self.robot.get_allscan()
            self._got_new_scan = True

    def update_reference_direction(
        self,
        laser_scan: np.ndarray = None,
        position: np.ndarray = None,
        in_robot_frame: bool = None,
        initial_velocity: np.ndarray = None,
    ) -> np.ndarray:

        if in_robot_frame is not None:
            self._laserscan_in_robot_frame = in_robot_frame

        if laser_scan is None:
            laser_scan = self.laserscan
        else:
            self.laserscan = laser_scan

        if laser_scan is None or len(laser_scan.shape) < 2 or not laser_scan.shape[1]:
            self.reference_direction = np.zeros(self.robot.pose.position.shape)
            return self.reference_direction
        # TODO: position is currently unused...
        # (
        #     laser_scan,
        #     ref_dirs,
        #     relative_distances,
        # ) = self.robot.get_relative_positions_and_dists(
        #     laser_scan, in_robot_frame=self._laserscan_in_robot_frame
        # )
        (laser_scan, ref_dirs, relative_distances,) = get_relative_positions_and_dists(
            center_position=position,
            control_radius=self.control_radius,
            datapoints=self.datapoints,
            in_local_frame=False,
        )

        self.weights = self.get_weight_from_distances(
            relative_distances, ref_dirs, initial_velocity
        )

        # (-1) or not ...
        self.reference_direction = (-1) * np.sum(
            ref_dirs * np.tile(self.weights, (ref_dirs.shape[0], 1)), axis=1
        )

        if self.evaluate_normal:
            self.update_normal_direction(laser_scan, self.weights, ref_dirs)

        # For Temporary plotting [remove after submission]
        if hasattr(self, "debug_mode") and self.debug_mode:
            warnings.warn("Storing refs and norms.")
            self.ref_dirs = (-1) * ref_dirs

        return self.reference_direction

    def get_weight_from_distances(
        self,
        distances: np.ndarray,
        directions: np.ndarray = None,
        initial_velocity: np.ndarray = None,
    ):
        """Returns an array of weights with the same dimensions as distances input."""
        # => get weighted evaluation along the robot
        # to obtain linear + angular velocity
        if np.sum(distances < self.margin_weight):
            warnings.warn("Treat the small-weight case.")

            distances = distances - np.min(distances) + self.margin_weight

        num_points = distances.shape[0]
        weights = (self.weight_factor / distances) ** self.weight_power
        self.distance_weight_sum = np.sum(weights)

        if (
            self.evaluate_velocity_weight
            and directions is not None
            and initial_velocity is not None
        ):
            weights = self.reduce_wake_effect(weights, initial_velocity, directions)

        if (
            self.weight_max_norm is not None
            and self.distance_weight_sum > self.weight_max_norm
        ):
            self.distance_weight_sum = self.weight_max_norm

        if self.distance_weight_sum > 1:
            return weights / self.distance_weight_sum
        else:
            return weights

    def update_normal_direction(self, laser_scan, weights, ref_dirs):
        """Update the normal direction and normal angle with resect to the reference."""
        # NOTE: This does not work very well [anymore] ?!

        norm_ref_dir = LA.norm(self.reference_direction)
        if not norm_ref_dir:
            return

        if len(weights) <= 1:
            self.normal_direction
            return

        # Only evaluate in presence of non-trivial reference direction
        if laser_scan.shape[0] != 2:
            raise NotImplementedError("Only done for ii>1")

        # Reduce data to necesarry ones only
        ind_nonzero = weights > 0

        weights = weights[ind_nonzero]
        ref_dirs = ref_dirs[:, ind_nonzero]
        laser_scan = laser_scan[:, ind_nonzero]

        tangents = laser_scan - np.roll(laser_scan, shift=1, axis=1)

        # normals = np.vstack(((-1)*tangents[1, :], tangents[0, :]))
        normals = np.vstack((tangents[1, :], (-1) * tangents[0, :]))

        # Remove any which happended through overlap / or other unexpected way
        ind_bad = np.sum(ref_dirs * normals, axis=0) < 0

        if any(ind_bad):
            normals[:, ind_bad] = ref_dirs[:, ind_bad]

        normals = normals / np.tile(LA.norm(normals, axis=0), (normals.shape[0], 1))

        # Average over two steps
        # normals = (normals + np.roll(normals, shift=1, axis=1)) / 2.0
        # normals = (normals + np.roll(normals, shift=(-1), axis=1)) / 2.0
        normals = normals / np.tile(LA.norm(normals, axis=0), (normals.shape[0], 1))

        # Evaluate normal offset
        norm_weights = LA.norm(weights)
        if not norm_weights:
            raise ValueError("Trivial value does not make sense.")

        if norm_weights != 1:
            weights = weights / norm_weights

        # Invert all, cause some weird mix
        normals = (-1) * normals
        ref_dirs = (-1) * ref_dirs

        # Average the weight here (!) -> since I cannot directly average the normal
        weights = (weights + np.roll(weights, shift=(-1))) / 2.0

        normal_offset = np.sum(
            np.tile(weights, (normals.shape[0], 1)) * (normals - ref_dirs), axis=1
        )

        ref_normalized = self.reference_direction / LA.norm(self.reference_direction)

        dot_prod = (-1) * np.dot(ref_normalized, normal_offset)

        if dot_prod <= np.sqrt(2) / 2:
            ref_factor = 1
        else:
            ref_factor = np.sqrt(2) * dot_prod

        # Because self.referencence_direction is pointing in the other direction than the ref-list
        # normal_offset *= (-1)

        # print("Normal direction", self.normal_direction)
        self.normal_direction = ref_factor * ref_normalized + normal_offset

        if not LA.norm(self.normal_direction):
            breakpoint()

        # Zero division test does (in theory) not have to be done
        self.normal_direction = self.normal_direction / LA.norm(self.normal_direction)

        # Temporary plotting
        if hasattr(self, "debug_mode") and self.debug_mode:
            warnings.warn("Storing refs and norms.")
            self.normal_dirs = normals
            self.ref_dirs = ref_dirs

    def old_part_of_normal(self):
        norm_angles = np.cross(ref_dirs, normals, axisa=0, axisb=0)
        ind_critical = np.abs(norm_angles) > np.sin(self.max_angle_ref_norm)
        if any(ind_critical):
            norm_angles[ind_critical] = np.copysign(
                np.sin(self.max_angle_ref_norm), norm_angles[ind_critical]
            )

        norm_dir = np.sum(norm_angles * weights)
        self.norm_angle = np.arcsin(self.norm_angle)

        # Add angle to reference direction
        unit_ref_dir = self.reference_direction / norm_ref_dir
        norm_angle += np.arctan2(unit_ref_dir[1], unit_ref_dir[0])

        self.normal_direction = np.array(
            [np.cos(self.norm_angle), np.sin(self.norm_angle)]
        )

        return self.normal_direction


class FastLidarAvoider(SampledAvoider):
    pass
