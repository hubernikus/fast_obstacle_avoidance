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
    def __init__(self, robot: ControlRobot) -> None:
        self.robot = robot

    def update_laserscan(self, laser_scan: np.ndarray) -> None:
        # self.laser_scan = laserscan
        self.reference_direction = self.get_reference_direction(
            laser_scan)
        
    def evaluate(self,
                 initial_velocity: np.ndarray, 
                 limit_velocity_magnitude: bool = True) -> None:
        """ Modulate velocity and return DS. """
        # For each control_points
        # -> get distance (minus radius)
        # -> get closest points
        # -> get_weights => how?
        ref_norm = LA.norm(self.reference_direction)

        if not ref_norm:
            # Not modulated when far away from everywhere
            return initial_velocity

        deomposition_matrix = get_orthogonal_basis(
            self.reference_direction/ref_norm, normalize=False)
        
        stretching_vector = self.get_stretching_vector(
            ref_norm, self.reference_direction, initial_velocity)
        
        modulated_velocity = deomposition_matrix.T @ initial_velocity
        modulated_velocity = np.diag(stretching_vector) @ modulated_velocity
        modulated_velocity = deomposition_matrix @ modulated_velocity

        if limit_velocity_magnitude:
            mod_norm = LA.norm(modulated_velocity)
            init_norm = LA.norm(initial_velocity)
            
            if mod_norm > init_norm:
                modulated_velocity = modulated_velocity * (init_norm/mod_norm)
            
        return modulated_velocity

    def get_stretching_vector(
        self, ref_norm: float,
        reference_direction: np.ndarray = None,
        initial_velocity: np.ndarray = None,
        tail_effect: bool = False) -> np.ndarray:
        weight = self.get_weight_from_norm(ref_norm)

        if np.dot(reference_direction, initial_velocity) < 0:
            normal_stretch = 1
        else:
            
            normal_stretch = 1 - weight
        
        stretching_vector = np.hstack((
            normal_stretch,
            1+weight * np.ones(reference_direction.shape[0]-1)
            ))
        
        return stretching_vector

    def get_reference_direction(self, laser_scan: np.ndarray) -> np.ndarray:
        relative_position, relative_distances = (
            self.robot.get_relative_positions_and_dists(laser_scan)
            )
        weights = self.get_weight_from_distances(relative_distances)

        reference_direction = np.sum(
            relative_position * np.tile(weights, (relative_position.shape[0], 1)),
            axis=1
            )

        # fig, ax = plt.subplots(figsize=(12, 6))
        # ax.plot(laser_scan[0, :], laser_scan[1, :], '.', color='k')
        # ax.plot(0, 0, 'o', color='r')

        # ax.quiver(laser_scan[0, :], laser_scan[1, :],
                  # relative_position[0, :]*weights,
                  # relative_position[1, :]*weights, color='r', scale=2)
        
        return reference_direction

    def get_weight_from_norm(self, norm):
        return norm
    
    def get_weight_from_distances(
        self, distances, weight_factor=0.1, weight_power=2.0, margin_weight=1e-3):
        # => get weighted evaluation along the robot
        # to obtain linear + angular velocity
        if any(distances < margin_weight):
            warnings.warn("Treat the small-weight case.")

            distances = distances - np.min(distances) + margin_weight

        weight = (1 / distances)**weight_power * weight_factor

        weight_sum = np.sum(weight)
        
        if weight_sum > 1:
            return weight / weight_sum
        else:
            weight
            
    def limit_velocity(self):
        pass

    def limit_acceleration(self):
        pass
