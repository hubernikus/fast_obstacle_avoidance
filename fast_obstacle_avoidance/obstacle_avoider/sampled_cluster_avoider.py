"""
Obstacle Avoider Dedicated to Lidar Data
"""

from __future__ import annotations

import warnings
from typing import Optional, Protocol
from enum import Enum, auto

from timeit import default_timer as timer

import math

import numpy as np
from numpy import linalg as LA

from sklearn.cluster import DBSCAN, KMeans

from vartools.linalg import get_orthogonal_basis
from vartools.directional_space import get_directional_weighted_sum
from vartools.vector_rotation import VectorRotationXd

from fast_obstacle_avoidance.control_robot import BaseRobot
from fast_obstacle_avoidance.directional_learning import DirectionalSoftKMeans

from .stretching_matrix import StretchingMatrixTrigonometric
from ._base import SingleModulationAvoider
from .lidar_avoider import SampledAvoider, get_relative_positions_and_dists


class Clusterer(Protocol):
    @property
    def labels_(self) -> np.ndarray:
        pass

    def fit(self, XX: np.ndarray) -> np.ndarray:
        pass


class ClustererType(Enum):
    DBSCAN = auto()
    DIRECTIONAL_SOFT_KMEANS = auto()


class SampledClusterAvoider:
    """
    Sub-divides the space of points into local clusters which are then avoided.

    # TODO:
    # - further automate weights
    # -
    """

    def __init__(
        self,
        robot: Optional[BaseRobot] = None,
        evaluate_normal: bool = False,
        cluster_params: dict = None,
        stretching_matrix=None,
        weight_max_norm: float = 1e5,
        weight_factor: float = 2 * np.pi / 10,
        weight_power: float = 2.0,
        control_radius: float = 1.0,
        clusterer: Clusterer | ClustererType = ClustererType.DIRECTIONAL_SOFT_KMEANS,
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

        if robot is None:
            self._laserscan_in_robot_frame = False
            if control_radius is None:
                raise Exception("control_radius is needed.")

            self.control_radius = control_radius
        else:
            self._laserscan_in_robot_frame = True
            self.control_radius = self.robot.control_radius

        if clusterer == ClustererType.DIRECTIONAL_SOFT_KMEANS:

            self.clusterer = DirectionalSoftKMeans(stiffness=5.0)
            print(
                f"Using DirectionalSoftKmeans with stiffness={self.clusterer.stiffness}"
            )

        elif clusterer == ClustererType.DBSCAN:
            # WARNING: DBSCAN has been observed to lead to 'fast' switching.
            # mabye it could be used with (fading) memory perception of the environment
            # However this would lead to increase in the computational cost
            if cluster_params is None:
                cluster_params = {"eps": 2 * self.control_radius, "min_samples": 3}
            self.clusterer = DBSCAN(**cluster_params)
            print(f"Using DBSCAN.")

        else:
            self.clusterer = clusterer

        self.sample_handler = SampledAvoider(
            self.robot,
            weight_max_norm=weight_max_norm,
            weight_factor=weight_factor,
            weight_power=weight_power,
            control_radius=self.control_radius,
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

    # @property
    # def control_radius(self) -> float:
    #     """Assumption of a nonzero radius, to not fall into the measurement gaps."""
    #     return self.robot.control_radius

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

        # Set centers
        self._cluster_centers = np.zeros((self.dimension, len(self.unique_labels)))
        for ii, label in enumerate(self.unique_labels):
            self._cluster_centers[:, ii] = np.mean(
                self._datapoints[:, label == self.clusterer.labels_], axis=1
            )

    def get_cluster_centers(self) -> np.ndarray:
        return self._cluster_centers

    def _cluster_close_outliers(self, position: np.ndarray) -> Optional(np.ndarray):
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

    def avoid(
        self, initial_velocity: np.ndarray, position: np.ndarray = None
    ) -> np.ndarray:
        if position is None:
            position = self.robot.pose.position

        velocity_norm = LA.norm(initial_velocity)
        if not velocity_norm:
            # Zero velocity -> no modulation needed
            return initial_velocity

        velocity_direction = initial_velocity / velocity_norm

        self._cluster_close_outliers(position)

        (laser_scan, ref_dirs, relative_distances,) = get_relative_positions_and_dists(
            center_position=position,
            control_radius=self.control_radius,
            datapoints=self.datapoints,
            in_local_frame=False,
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
        self.modulated_velocities = np.zeros((self.dimension, self.n_obstacles))

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

            self.modulated_velocities[:, ii] = self.modulate(
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
                self.modulated_velocities = np.append(
                    self.modulated_velocities,
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

        self.normalized_weights = global_weights / weight_sum
        velocity = get_directional_weighted_sum(
            null_direction=velocity_direction,
            weights=self.normalized_weights,
            directions=self.modulated_velocities,
        )

        max_weight = np.max(global_weights)
        if max_weight > 0:
            averaged_normal = np.sum(
                self.normal_directions
                * np.tile(self.normalized_weights, (self.dimension, 1)),
                axis=1,
            )
            min_distance = np.min(relative_distances)
            scaling = self.compute_velocity_scaling(
                initial_velocity,
                averaged_normal,
                min_distance,
                max_distance=2 * self.control_radius + 1,
            )
            # print("scaling", scaling)
            # print("min_distance", min_distance)
        else:
            scaling = 1.0

        return velocity * velocity_norm * scaling

    def compute_velocity_scaling(
        self,
        velocity: np.ndarray,
        normal: np.ndarray,
        distance: float,
        dot_scaling: float = 0.8,
        power_root: float = 3,
        max_distance: float = 1e6,
    ) -> float:
        # TODO: very similar to 'compute_safe_magnitude' -> can they be merged (?!)
        if not (rotated_norm := np.linalg.norm(velocity)):
            return 1.0

        dot_product = np.dot(velocity, normal)
        normal_norm = np.linalg.norm(normal)

        if dot_product > 0 or np.isclose(normal_norm, 0):
            return 1.0

        elif distance <= 0.0:
            if dot_product < (-1e-3):
                return 0.0
            return 1.0

        elif distance >= max_distance:
            return 1.0

        # At this stage, the dot product is negative
        power_factor = ((max_distance - distance) / (distance) * normal_norm) ** (
            1.0 / power_root
        )
        return ((1.0 + dot_scaling * dot_product)) ** power_factor

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

        modulated_velocity = (
            decomposition_matrix
            @ stretching_matrix
            @ LA.pinv(decomposition_matrix)
            @ velocity
        )

        return modulated_velocity

    def limit_reference_from_offset(self, normal_direction, reference_direction):
        """The offset of the reference direction is limited to ensure convergence."""
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

        max_angle = 0.45 * np.pi
        if vector_rot.rotation_angle > max_angle:
            vector_rot.rotation_angle = (
                (np.pi - vector_rot.rotation_angle) / (math.pi - max_angle) * max_angle
            )

        # vector_rot.rotation_angle = vector_rot.rotation_angle * weight
        return vector_rot.rotate(normal_direction, weight)
