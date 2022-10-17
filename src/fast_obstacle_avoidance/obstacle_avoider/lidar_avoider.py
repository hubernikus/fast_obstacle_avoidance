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

from sklearn.cluster import DBSCAN, KMeans

from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum

from dynamic_obstacle_avoidance.rotational.vector_rotation import VectorRotationXd

from fast_obstacle_avoidance.control_robot import BaseRobot

from ._base import SingleModulationAvoider
from .stretching_matrix import StretchingMatrixTrigonometric


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

        self._laserscan_in_robot_frame = True

    # @property
    # def norm_angle(self):
    # return np.

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
        (
            laser_scan,
            ref_dirs,
            relative_distances,
        ) = self.robot.get_relative_positions_and_dists(
            laser_scan, in_robot_frame=self._laserscan_in_robot_frame
        )

        self.weights = self.get_weight_from_distances(relative_distances)

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
        weights = (1 / distances) ** self.weight_power * (self.weight_factor)

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


class SampledClusterAvoider:
    """
    Sub-divides the space of points into local clusters which are then avoided.

    # TODO:
    # - further automate weights
    # -
    """

    def __init__(
        self,
        robot: BaseRobot,
        evaluate_normal: bool = False,
        cluster_params: dict = None,
        stretching_matrix=None,
        weight_max_norm=1e5,
        weight_factor=2 * np.pi / 10,
        weight_power=2.0,
        # delta_sampling: float = delta_sampling
        # *args,
        # **kwargs,
    ) -> None:
        self.robot = robot

        self.evaluate_normal = evaluate_normal
        self.max_angle_ref_norm = 80 * np.pi / 180

        # For the moment, delta_sampling is not used
        # self.delta_sampling = delta_sampling

        # super().__init__(*args, **kwargs)

        if stretching_matrix is None:
            self.stretching_matrix = StretchingMatrixTrigonometric()
            # self.stretching_matrix.free_tail_flow = False
        else:
            self.stretching_matrix = stretching_matrix

        self._laserscan_in_robot_frame = True

        if cluster_params is None:
            cluster_params = {"eps": 2 * self.robot.control_radius, "min_samples": 3}

        self.clusterer = DBSCAN(**cluster_params)

        self.sample_handler = SampledAvoider(
            self.robot,
            weight_max_norm=weight_max_norm,
            weight_factor=weight_factor,
            weight_power=weight_power,
        )

    @classmethod
    def from_kmeans(cls, *args, **kwargs) -> SampledClusterAvoider:
        new_inst = cls(*args, **kwargs)

        # Overwrite clusterer
        # Alternative way to do the avoidance
        new_inst.clusterer = KMeans(n_clusters=new_inst.dimensions**2)

        raise NotImplementedError("The class is not working with KMeans yet.")
        return new_inst

    @property
    def dimension(self) -> int:
        return self.datapoints.shape[0]

    @property
    def control_radius(self) -> float:
        """Assumption of a nonzero radius, to not fall into the measurement gaps."""
        return self.robot.control_radius

    @property
    def datapoints(self) -> np.ndarray:
        # Property to make consistent with mixed avoider.
        # But change this variable name in the future
        return self._datapoints

    @datapoints.setter
    def datapoints(self, value: np.ndarray):
        """Returns global datapoints"""
        self._datapoints = value

    @property
    def n_obstacles(self) -> int:
        return self._cluster_centers.shape[1]

    def center_positions(self) -> np.ndarray:
        return np.hstack((self._center_positions, self._center_outliers))

    def update_reference_direction(self, *args, **kwargs) -> None:
        return self.update_sample_points(*args, **kwargs)

    def update_laserscan(self, *args, **kwargs) -> None:
        return self.update_sample_points(*args, **kwargs)

    def update_sample_points(
        self, datapoints: np.ndarray = None, in_robot_frame: bool = True
    ) -> None:
        if in_robot_frame:
            self._datapoints = self.robot.pose.transform_from_relative(datapoints)
        else:
            self._datapoints = datapoints

        start = timer()
        self.clusterer.fit(self._datapoints.T)
        end = timer()
        print(f"Clustering time {round((end - start)*1000, 2)} ms.")

        self.unique_labels = np.unique(self.clusterer.labels_)
        self.unique_labels = np.delete(self.unique_labels, self.unique_labels == -1)

        self._cluster_centers = np.zeros((self.dimension, len(self.unique_labels)))

        for ii, label in enumerate(self.unique_labels):
            self._cluster_centers[:, ii] = np.mean(
                self._datapoints[:, label == self.clusterer.labels_], axis=1
            )

    def _cluster_close_outliers(self, position) -> Optional(np.ndarray):
        # TODO: check if there are outliers closer than the closest cluster
        # TODO: should you ever redo the clustering (?)

        # Check if there are points really close...
        # (just looking at centers is unfortunately not enough, since a cluster could go around the robot)
        clustered_points = self._datapoints[:, self.clusterer.labels_ != -1]

        dist_closest_cluster = np.min(
            LA.norm(
                # self._center_positions
                # - np.tile(position, (self._cluster_centers.shape[1], 1)).T,
                clustered_points - np.tile(position, (clustered_points.shape[1], 1)).T,
                axis=0,
            )
        )

        ind_outliers = self.clusterer.labels_ == -1
        outliers = self._datapoints[:, ind_outliers]
        distances = LA.norm(
            outliers - np.tile(position, (outliers.shape[1], 1)).T, axis=0
        )
        ind_close = distances < dist_closest_cluster

        if not np.sum(ind_close):
            self._ind_close_outliers = np.zeros(0)
            self._close_outliers = np.zeros((self.dimension, 0))
            self._center_outliers = np.zeros((self.dimension, 0))
            return

        # TODO: if the outliers lie too badly and too close, they could be split (?)
        tmp_outliers = distances < dist_closest_cluster
        self._ind_close_outliers = np.arange(ind_outliers.shape[0])[ind_outliers][
            tmp_outliers
        ]
        self._close_outliers = outliers[:, tmp_outliers]
        self._center_outliers = np.array(np.mean(self._close_outliers, axis=1)).reshape(
            self.dimension, -1
        )

        center_dir = np.sum(self._close_outliers, axis=1)
        if not LA.norm(center_dir):
            warnings.warn("TODO: Investigate clustering of the close-outliers.")

        if np.sum(center_dir.T @ self._close_outliers):
            warnings.warn("TODO: Investigate clustering of the close-outliers.")

    def avoid(self, velocity: np.ndarray, position: np.ndarray = None) -> np.ndarray:
        if position is None:
            position = self.robot.pose.position

        velocity_norm = LA.norm(velocity)
        if not velocity_norm:
            # Zero velocity -> no modulation needed
            return velocity

        velocity_direction = velocity / velocity_norm

        self._cluster_close_outliers(position)

        (
            laser_scan,
            ref_dirs,
            relative_distances,
        ) = self.robot.get_relative_positions_and_dists(
            self.datapoints, in_robot_frame=False
        )

        local_weights = self.sample_handler.get_weight_from_distances(
            relative_distances,
            initial_velocity=velocity_direction,
            directions=ref_dirs,
        )

        # The reference / normal is only store for nice visualization
        self.reference_directions = np.zeros((self.dimension, self.n_obstacles))
        self.normal_directions = np.zeros((self.dimension, self.n_obstacles))

        global_weights = np.zeros(self.n_obstacles)
        modulated_velocities = np.zeros((self.dimension, self.n_obstacles))

        # For all cluster
        for ii in range(self.n_obstacles):
            ind_cluster = ii == self.clusterer.labels_

            normal_direction = (-1) * np.sum(
                ref_dirs[:, ind_cluster]
                * np.tile(local_weights[ind_cluster], (self.dimension, 1)),
                axis=1,
            )

            normal_norm = LA.norm(normal_direction)
            normal_direction = normal_direction / normal_norm
            global_weights[ii] = normal_norm

            reference_direction = position - self._cluster_centers[:, ii]
            reference_direction = self.limit_reference_from_offset(
                normal_direction, reference_direction
            )

            basis_matrix = get_orthogonal_basis(normal_direction)
            basis_matrix[:, 0] = reference_direction

            stretch_matrix = self.stretching_matrix.get(
                normal_norm,
                reference_direction,
                normal_direction,
                velocity_direction,
            )

            modulated_velocities[:, ii] = self.modulate(
                velocity_direction,
                decomposition_matrix=basis_matrix,
                stretching_matrix=stretch_matrix,
            )

            self.reference_directions[:, ii] = reference_direction
            self.normal_directions[:, ii] = normal_direction

        # Check if there is outliers -> additionally observe them
        if len(self._ind_close_outliers):
            local_weights = self.sample_handler.get_weight_from_distances(
                relative_distances[self._ind_close_outliers]
            )

            reference_direction = (-1) * np.sum(
                ref_dirs[:, self._ind_close_outliers]
                * np.tile(local_weights, (self.dimension, 1)),
                axis=1,
            )

            if ref_norm := LA.norm(reference_direction):
                reference_direction = reference_direction / ref_norm
                global_weights = np.append(global_weights, [ref_norm])

                # Basis matrix is orthogonal for the close points
                basis_matrix = get_orthogonal_basis(reference_direction)
                stretch_matrix = self.stretching_matrix.get(
                    ref_norm,
                    reference_direction,
                    initial_velocity=velocity_direction,
                )

                mod_vel = self.modulate(
                    velocity_direction,
                    decomposition_matrix=basis_matrix,
                    stretching_matrix=stretch_matrix,
                )
                modulated_velocities = np.append(
                    modulated_velocities,
                    mod_vel.reshape(self.dimension, 1),
                    axis=1,
                )

                self.reference_directions = np.append(
                    self.reference_directions,
                    reference_direction.reshape(self.dimension, 1),
                    axis=1,
                )
                self.normal_directions = np.append(
                    self.normal_directions,
                    normal_direction.reshape(self.dimension, 1),
                    axis=1,
                )
            else:
                warnings.warn("Zero length reference.")

        if not (weight_sum := np.sum(global_weights)):
            return velocity

        velocity = get_directional_weighted_sum(
            null_direction=velocity_direction,
            weights=global_weights / weight_sum,
            directions=modulated_velocities,
        )

        return velocity * velocity_norm

    def modulate(
        self,
        velocity,
        decomposition_matrix,
        stretching_matrix,
        tail_effect: bool = True,
    ):
        """Modulate with tail-effect."""
        # local_velocity = LA.pinv(decomposition_matrix) @ velocity

        #     if tail_effect and local_velocity[0] > 0:
        #         # Already moving away
        #         weight = local_velocity[0] / LA.norm(velocity)
        #         new_eigenvalues = weight + (1 - weight) * stretching_matrix[0, 0]

        #         stretching_matrix = np.eye(stretching_matrix.shape[0]) * new_eigenvalues
        #     breakpoint()

        return (
            decomposition_matrix
            @ stretching_matrix
            @ LA.pinv(decomposition_matrix)
            @ velocity
        )

    def limit_reference_from_offset(self, normal_direction, reference_direction):
        """Get reference room."""
        # reference_direction = position - self._cluster_centers[:, ii]

        if not (ref_norm := LA.norm(reference_direction)):
            return normal_direction

        reference_direction = reference_direction / ref_norm
        weight = min(ref_norm / self.control_radius, 1)
        if not weight:
            return normal_direction

        vector_rot = VectorRotationXd.from_directions(
            normal_direction, reference_direction
        )

        max_ang = 0.45 * np.pi
        if vector_rot.rotation_angle > max_ang:
            vector_rot.rotation_angle = np.pi - vector_rot.rotation_angle

        # vector_rot.rotation_angle = vector_rot.rotation_angle * weight
        return vector_rot.rotate(normal_direction, weight)
