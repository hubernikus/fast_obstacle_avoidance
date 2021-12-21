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

from ._base import SingleModulationAvoider


class FastObstacleAvoider(SingleModulationAvoider):
    def __init__(self, obstacle_environment):
        """Initialize with obstacle list"""
        self.obstacle_environment = obstacle_environment

        super().__init__()

        if self.obstacle_environment.dimension == 3:
            from scipy.spatial.transform import Rotation as R

    def update_reference_direction(self, position):
        norm_dirs = np.zeros(
            (self.obstacle_environment.dimension, self.obstacle_environment.n_obstacles)
        )
        ref_dirs = np.zeros(norm_dirs.shape)
        relative_distances = np.zeros((norm_dirs.shape[1]))

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
