"""
Mixed Environments from Lidar & Obstacle Detection Data
"""
import warnings

import numpy as np
from numpy import linalg as LA

from dynamic_obstacle_avoidance.obstacles import CircularObstacle

from ._base import SingleModulationAvoider
from .obstacle_avoider import FastObstacleAvoider
from .lidar_avoider import FastLidarAvoider


class MixedEnvironmentAvoider(SingleModulationAvoider):
    """Mixed Environments"""

    def __init__(
        self,
        robot,
        evaluate_normal=True,
        recompute_all=True,
        delta_sampling=1.0,
        scaling_laserscan_weight=1.0,
        scaling_obstacle_weight=1.0,
        *args,
        **kwargs,
    ):
        """
        Arguments
        ----------
        recompute_all: Since the code is not (throughfully) tested, we just
            compute everything. In the future this should be fixed by doing a state
            machine analysis etc.
        """
        self.recompute_all = recompute_all
        super().__init__(*args, **kwargs)

        self._robot = robot

        # One for obstacles one for environments
        self.lidar_avoider = FastLidarAvoider(
            self.robot, evaluate_normal=False, weight_factor=delta_sampling
        )
        self.obstacle_avoider = FastObstacleAvoider(
            self.robot.obstacle_environment, robot=self.robot
        )

        self.evaluate_normal = evaluate_normal

        self.scaling_laserscan_weight = scaling_laserscan_weight
        self.scaling_obstacle_weight = scaling_obstacle_weight

        self._got_new_scan = True

        self.laserscan = None

    @property
    def robot(self):
        # This must not be changed, otherwise the linkage is lost
        return self._robot

    @property
    def datapoints(self):
        return self.laserscan

    @datapoints.setter
    def datapoints(self, value):
        self.laserscan = value

    @property
    def obstacle_environment(self):
        return self.robot.obstacle_environment

    @property
    def dimension(self):
        return self.obstacle_avoider.obstacle_environment.dimension

    def update_laserscan(self, laserscan=None, in_robot_frame=True):
        if in_robot_frame is False:
            raise NotImplementedError()

        if laserscan is not None:
            self.laserscan = laserscan
            self._got_new_scan = True

        elif self.robot.has_newscan:
            self.laserscan = self.robot.get_allscan()
            self._got_new_scan = True

    def update_reference_direction(
        self, laserscan=None, position=None, in_robot_frame=True
    ):
        """Clean up lidar first (remove inside obstacles)
        and then get reference, once for the avoider"""
        if laserscan is not None:
            self.update_laserscan(laserscan)

        if not self.recompute_all and (
            self._got_new_scan and not self.robot.has_new_obstacles
        ):
            # Nothing as changed - keep existing laserscan
            return self.reference_direction

        if self.recompute_all or self._got_new_scan:
            cleanscan = self.get_scan_without_ocluded_points()
            self.lidar_avoider.update_laserscan(cleanscan)

            self.lidar_avoider.update_reference_direction(
                position=position,
                in_robot_frame=in_robot_frame,
            )

        if self.recompute_all or self.robot.has_new_obstacles:
            self.obstacle_avoider.update_reference_direction(
                position=position, in_robot_frame=in_robot_frame
            )

        self.weights = self.get_mixed_weights()

        self.reference_direction = (
            self.weights[0] * self.lidar_avoider.reference_direction
            + self.weights[1] * self.obstacle_avoider.reference_direction
        )

        if self.lidar_avoider.relative_velocity is not None:
            raise NotImplementedError("Not implemented for relative lidar velocity.")

        # Update velocity
        if self.obstacle_avoider.relative_velocity is not None:
            self.relative_velocity = (
                self.weights[1] * self.obstacle_avoider.relative_velocity
            )

        # Potentially update normal direction
        if self.evaluate_normal:
            self.update_normal_direction(self.weights)

        return self.reference_direction

    def update_normal_direction(self, weights):
        """Normal direction update is simplified to environment
        where only one has the actual normal."""
        if self.lidar_avoider.normal_direction is not None and LA.norm(
            self.lidar_avoider.normal_direction
        ):
            raise NotImplementedError()

        if self.obstacle_avoider.normal_direction is None or not LA.norm(
            self.obstacle_avoider.normal_direction
        ):
            self.normal_direction = self.reference_direction
            return

        delta_normal = self.obstacle_avoider.normal_direction - self.reference_direction
        self.normal_direction = (
            self.reference_direction + self.obstacle_weight * delta_normal
        )

    def get_scan_without_ocluded_points(self):
        """Remove laserscan which is within boundaries of obstacles
        and then update lidar-reference direction."""
        self.update_laserscan()
        laserscan = np.copy(self.laserscan)

        for obs in self.obstacle_avoider.obstacle_environment:
            if isinstance(obs, CircularObstacle) or (
                hasattr(obs, "is_human") and obs.is_human
            ):
                # Get gamma from array for circular obstacles only (!)
                dirs = laserscan - np.tile(obs.position, (laserscan.shape[1], 1)).T
                gamma_vals = LA.norm(dirs, axis=0) - obs.radius

                is_outside = gamma_vals > 0
            else:
                gamma_vals = np.zeros(laserscan.shape[1])
                for ii in range(laserscan.shape[1]):
                    gamma_vals[ii] = obs.get_gamma(
                        laserscan[:, ii], in_global_frame=True
                    )
                is_outside = gamma_vals > 1

            if not np.sum(is_outside):
                return np.zeros((self.dimension, 0))

            laserscan = laserscan[:, is_outside]
        return laserscan

    @property
    def sample_weight(self):
        return self.weights[0]

    @property
    def obstacle_weight(self):
        return self.weights[1]

    def get_mixed_weights(self, max_weight=1e9):
        """Calculate the weights from the mixed environments."""
        weights = np.array(
            [
                self.lidar_avoider.distance_weight_sum,
                self.obstacle_avoider.distance_weight_sum,
            ]
        )

        # Remove none-values
        none_values = np.array([(ww is None) for ww in weights])
        if any(none_values):
            weights[none_values] = 0

        # All zero values
        if not np.sum(weights):
            return weights

        ind_max = weights >= 1
        if any(ind_max):
            weights = np.zeros(ind_max.shape)
            weights[ind_max] = max_weight
            weights[~ind_max] = 1 / (1 - weights[~ind_max])
        else:
            weights = 1 / (1 - weights)
            weights = np.minimum(weights, max_weight)

        # Do scaling of laserscan weight
        scaling = np.array(
            [self.scaling_laserscan_weight, self.scaling_obstacle_weight]
        )

        # scaling = scaling / np.sum(scaling)
        weights = np.array(weights) * scaling

        if np.sum(weights) > 1:
            weights = weights / np.sum(weights)

        return weights
