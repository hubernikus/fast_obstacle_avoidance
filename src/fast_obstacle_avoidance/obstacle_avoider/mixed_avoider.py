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
    def __init__(self,
                 robot,
                 evaluate_normal=False, 
                 recompute_all=True,
                 scaling_laserscan_weight=1.0,
                 *args, **kwargs):
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
            self.robot, evaluate_normal=False)
        self.obstacle_avoider = FastObstacleAvoider(
            self.robot.obstacle_environment, robot=self.robot
        )

        self.evaluate_normal = evaluate_normal
        
        self.scaling_laserscan_weight = scaling_laserscan_weight
        
        self._got_new_scan = True

        self.laserscan = None

    @property
    def robot(self):
        # This must not be changed, otherwise the linkage is lost
        return self._robot

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

    def update_reference_direction(self, laserscan=None, in_robot_frame=True):
        """Clean up lidar first (remove inside obstacles)
        and then get reference, once for the avoider"""
        if laserscan is not None:
            self.update_laserscan(laserscan)

        if (not self.recompute_all and
            (self._got_new_scan and not self.robot.has_new_obstacles)):
            # Nothing as changed - keep existing laserscan
            return self.reference_direction

        if (self.recompute_all or
            self._got_new_scan):
            cleanscan = self.get_scan_without_ocluded_points()
            self.lidar_avoider.update_laserscan(cleanscan)
            
            self.lidar_avoider.update_reference_direction(in_robot_frame=in_robot_frame)

        if (self.recompute_all or
            self.robot.has_new_obstacles):
            self.obstacle_avoider.update_reference_direction(
                in_robot_frame=in_robot_frame
            )

        weights = [
            self.lidar_avoider.distance_weight_sum,
            self.obstacle_avoider.distance_weight_sum,
        ]

        if np.sum(weights) > 1:
            weights = weights / np.sum(weights)

        weights[0] = weights[0] * self.scaling_laserscan_weight
        self.reference_direction = (
            weights[0] * self.lidar_avoider.reference_direction
            + weights[1] * self.obstacle_avoider.reference_direction
        )
        
        if self.lidar_avoider.relative_velocity is not None:
            raise NotImplementedError("Not implemented for relative lidar velocity.")

        # Update velocity
        if self.obstacle_avoider.relative_velocity is not None:
            self.relative_velocity = (
                weights[1] * self.obstacle_avoider.relative_velocity
            )

        # Potentially update normal direction
        if self.evaluate_normal:
            # TODO:
            raise NotImplementedError("This needs to be done again.")
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
            if (not isinstance(obs, CircularObstacle)
                and (not hasattr(obs, "is_human") or not obs.is_human)):
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
