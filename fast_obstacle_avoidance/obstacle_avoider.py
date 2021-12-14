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

import matplotlib.pyplot as plt

from fast_obstacle_avoidance.control_robot import ControlRobot

from vartools.linalg import get_orthogonal_basis


class FastObstacleAvoider:
    """ To proof:
    -> reference direction becomes normal (!) when getting very close (!)
    -> obstacle is being avoided when very close(!) [reference -> normal]
    -> No local minima (and maybe even convergence to attractor)
    -> add slight repulsion along the normal direction (when getting closer)
    """
    def __init__(self, robot: ControlRobot):
        self.robot = robot

    def evaluate(self, initial_velocity: np.ndarray, laser_scan: np.ndarray):
        """ Modulate velocity and return DS. """
        # For each control_points
        # -> get distance (minus radius)
        # -> get closest points
        # -> get_weights => how?
        reference_direction = self.get_reference_direction(laser_scan)

        ref_norm = LA.norm(reference_direction)

        if not ref_norm:
            # Not modulated when far away from everywhere
            return initial_velocity

        deomposition_matrix = get_orthogonal_basis(reference_direction/ref_norm, normalize=False)
        
        weight = self.get_weight_from_norm(ref_norm)
        stretching_vector = np.hstack((
            1-weight,
            1+weight * np.ones(reference_direction.shape[0]-1)
            ))

        modulated_velocity = deomposition_matrix.T @ initial_velocity
        modulated_velocity = np.diag(stretching_vector) @ modulated_velocity
        modulated_velocity = deomposition_matrix @ modulated_velocity
        
        return modulated_velocity

    def get_reference_direction(self, laser_scan: np.ndarray) -> np.ndarray:
        relative_position, relative_distances = (
            self.robot.get_relative_positions_and_dists(laser_scan)
            )
        weights = self.get_weight_from_distances(relative_distances)

        reference_direction = np.sum(
            relative_position * np.tile(weights, (relative_position.shape[0], 1)),
            axis=1
            )
        
        return reference_direction

    def get_weight_from_norm(self, norm):
        return norm
    
    def get_weight_from_distances(self, distances, weight_factor=1.0, margin_weight=1e-3):
        # => get weighted evaluation along the robot
        # to obtain linear + angular velocity
        if any(distances < margin_weight):
            warnings.warn("Treat the small-weight case.")

            distances = distances - np.min(distances) + margin_weight
        weight = 1 / distances * weight_factor

        weight_sum = np.sum(weight)
        
        if weight_sum > 1:
            return weight / weight_sum
        else:
            weight
            
    def limit_velocity(self):
        pass

    def limit_acceleration(self):
        pass
