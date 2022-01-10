"""
Obstacle Avoider Dedicated to Lidar Data
"""

import warnings

import numpy as np
from numpy import linalg as LA

from vartools.linalg import get_orthogonal_basis

from fast_obstacle_avoidance.control_robot import BaseRobot
from ._base import SingleModulationAvoider


class FastLidarAvoider(SingleModulationAvoider):
    """To proof:
    -> reference direction becomes normal (!) when getting very close (!)
    -> obstacle is being avoided when very close(!) [reference -> normal]
    -> No local minima (and maybe even convergence to attractor)
    -> add slight repulsion along the normal direction (when getting closer)
    """

    def __init__(self, robot: BaseRobot, evaluate_normal: bool = False) -> None:
        self.robot = robot

        self.evaluate_normal = evaluate_normal
        self.max_angle_ref_norm = 80 * np.pi / 180
        super().__init__()

    def update_reference_direction(
        self, laser_scan: np.ndarray = None, in_robot_frame: bool = True
    ) -> np.ndarray:
        if laser_scan is None:
            laser_scan = self.laser_scan

        (
            laser_scan,
            ref_dirs,
            relative_distances,
        ) = self.robot.get_relative_positions_and_dists(
            laser_scan, in_robot_frame=in_robot_frame
        )

        weights = self.get_weight_from_distances(relative_distances)

        self.reference_direction = (-1) * np.sum(
            ref_dirs * np.tile(weights, (ref_dirs.shape[0], 1)), axis=1
        )

        if self.evaluate_normal:
            self.update_normal_direction(laser_scan, weights, ref_dirs)

        print("ref dir", self.reference_direction)
        return self.reference_direction

    def update_normal_direction(self, laser_scan, weights, ref_dirs):
        """Update the normal direction and normal angle with resect to the reference."""
        norm_ref_dir = LA.norm(self.reference_direction)
        if not norm_ref_dir:
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

        normals = np.vstack(((-1) * tangents[1, :], tangents[0, :]))

        # Remove any which happended through overlap
        # Matrix dot-product
        ind_bad = np.sum(ref_dirs * normals, axis=0) < 0

        if any(ind_bad):
            normals[:, ind_bad] = ref_dirs[:, ind_bad]

        normals = normals / np.tile(LA.norm(normals, axis=0), (normals.shape[0], 1))

        # Average over two steps
        normals = (normals + np.roll(normals, shift=1, axis=1)) / 2.0
        normals = normals / np.tile(LA.norm(normals, axis=0), (normals.shape[0], 1))

        norm_angles = np.cross(ref_dirs, normals, axisa=0, axisb=0)
        ind_critical = np.abs(norm_angles) > np.sin(self.max_angle_ref_norm)
        if any(ind_critical):
            norm_angles[ind_critical] = np.copysign(
                np.sin(self.max_angle_ref_norm), norm_angles[ind_critical]
            )

        self.norm_angle = np.sum(norm_angles * weights)
        self.norm_angle = np.arcsin(norm_angle)

        # Add angle to reference direction
        unit_ref_dir = self.reference_direction / norm_ref_dir
        norm_angle += np.arctan2(unit_ref_dir[1], unit_ref_dir[0])

        self.normal_direction = np.array(
            [np.cos(self.norm_angle), np.sin(self.norm_angle)]
        )

        return self.normal_direction
        # print('devi', np.arcsin(np.cross(self.reference_direction,
        # self.normal_direction))
        # )