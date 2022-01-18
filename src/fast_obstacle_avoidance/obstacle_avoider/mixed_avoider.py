"""
Mixed Environments from Lidar & Obstacle Detection Data
"""
import numpy as np
from numpy import linalg as LA

from dynamic_obstacle_avoidance.obstacles import CircularObstacle

from ._base import SingleModulationAvoider
from .obstacle_avoider import FastObstacleAvoider
from .lidar_avoider import FastLidarAvoider


class MixedEnvironmentAvoider(SingleModulationAvoider):
    """Mixed Environments"""

    def __init__(self, robot, evaluate_normal=False):
        super().__init__()

        self.robot = robot
        # One for obstacles one for environments
        self.lidar_avoider = FastLidarAvoider(self.robot, evaluate_normal)
        self.obstacle_avoider = FastObstacleAvoider(
            self.robot.obstacle_environment, robot=self.robot)

        self.evaluate_normal = evaluate_normal

        self._got_new_scan = True

    @property
    def dimension(self):
        return self.obstacle_avoider.obstacle_environment.dimension

    def update_laserscan(self, laserscan=None):
        if laserscan is not None:
            self.laserscan = laserscan
            self._got_new_scan = True

        elif self.robot.has_newscan:
            self.laserscan = self.robot.get_allscan()
            self._got_new_scan = True

    def update_reference_direction(self, in_robot_frame=True):
        """Clean up lidar first (remove inside obstacles)
        and then get reference, once for the avoider."""
        if not self._got_new_scan and not self.robot.has_new_obstacles:
            # Nothing as changed - keep existing laserscan
            return self.reference_direction

        if self._got_new_scan:
            cleanscan = self.get_scan_without_ocluded_points()
            self.lidar_avoider.update_reference_direction(
                cleanscan, in_robot_frame=in_robot_frame
            )

        if self.robot.has_new_obstacles:
            self.obstacle_avoider.update_reference_direction(
                in_robot_frame=in_robot_frame
            )

        weights = [
            self.lidar_avoider.distance_weight_sum,
            self.obstacle_avoider.distance_weight_sum,
        ]
        
        if np.sum(weights) > 1:
            weights = weights / np.sum(weights)

        self.reference_direction = (
            weights[0] * self.lidar_avoider.reference_direction
            + weights[1] * self.obstacle_avoider.reference_direction
        )

        if self.lidar_avoider.relative_velocity is not None:
            raise NotImplementedError("Not implemented for relative lidar velocity.")
        
        # Update velocity
        if self.obstacle_avoider.relative_velocity is not None:
            self.relative_velocity = (
                weights[1]*self.obstacle_avoider.relative_velocity
                )
            
        # Potentially update normal direction
        if self.evaluate_normal:
            ref_norm = LA.norm(self.reference_direction)
            if ref_norm:
                if self.obstacle_avoider.obstacle_environment.dimension == 2:
                    self.norm_angle = (
                        weights[0] * self.lidar_avoider.norm_angle
                        + weights[1] * self.obstacle_avoider.norm_angle
                    )
                    self.normal_direction = np.array(
                        [np.cos(self.norm_angle), np.sin(self.norm_angle)]
                    )

                    unit_ref_dir = self.reference_direction / LA.norm(
                        self.reference_direction
                    )
                    self.norm_angle += np.arctan2(unit_ref_dir[1], unit_ref_dir[0])

                    self.normal_direction = np.array(
                        [np.cos(self.norm_angle), np.sin(self.norm_angle)]
                    )

                else:
                    raise NotImplementedError(
                        f"Not implemented for d={self.obstacle_avoider.obstacle_environment.dimension}"
                    )

        return self.reference_direction

    def get_scan_without_ocluded_points(self):
        """Remove laserscan which is within boundaries of obstacles
        and then update lidar-reference direction."""
        # TODO (!) -> ukpdate (circular) obstacle class for fast evaluation
        self.update_laserscan()
        laserscan = np.copy(self.laserscan)

        for obs in self.obstacle_avoider.obstacle_environment:
            if not hasattr(obs, "is_human") or not obs.is_human:
                raise NotImplementedError(
                    "Only implemented for non-circular  obstacles."
                )

            # Get gamma from array for circular obstacles only (!)
            dirs = laserscan - np.tile(obs.position, (laserscan.shape[1], 1)).T
            gamma_vals = LA.norm(dirs, axis=0) - obs.radius

            is_outside = gamma_vals > 0

            if not np.sum(is_outside):
                return np.zeros((self.dimension, 0))

            laserscan = laserscan[:, is_outside]

        return laserscan
