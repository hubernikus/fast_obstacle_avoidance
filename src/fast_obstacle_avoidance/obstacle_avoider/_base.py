"""
Simplified obstacle avoidance for mobile for mobile robots based on
DS & reference_direction

The algorithm is developed for shared control,
> human will know the best path, just be fast and don't get stuck
(make unexpected mistakes)
"""
# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

import warnings
from abc import ABC, abstractmethod

import numpy as np
from numpy import linalg as LA

from vartools.linalg import get_orthogonal_basis

from .stretching_matrix import StretchingMatrixBasic
from .stretching_matrix import StretchingMatrixTrigonometric
from .stretching_matrix import StretchingMatrixExponential


class SingleModulationAvoider(ABC):
    """
    Encapsulated the modulation in a single (virtual) obstacle

    Attributes
    ----------
    get_stretching_matrix: -> has 'get' attribute
    returns the stretching matrix based on the input f
    """

    def __init__(
        self,
        stretching_matrix: StretchingMatrixFunctor = None,
        # Parameters for the weight evaluation
        weight_max_norm: float = None,
        weight_factor: float = 10,
        weight_power: float = 1,
        margin_weight: float = 1e-3
    ):
        if stretching_matrix is None:
            self.stretching_matrix = StretchingMatrixTrigonometric()
        else:
            self.stretching_matrix = stretching_matrix

        self.reference_direction = None
        # TODO: initialize as np.zeros(dim), but for now this is easier deugging!

        # Normal direction (for inverted modulation)
        self.normal_direction = None

        # Normal angle -> can be useful for further calculation
        self.norm_angle = None

        # Distance weight sum (before normalization):
        # this is used for in the mixed environments
        self.distance_weight_sum = None

        # The maximum possible distance weight sum
        self.weight_max_norm = weight_max_norm
        self.weight_factor = weight_factor
        self.weight_power = weight_power
        self.margin_weight = margin_weight

        self.relative_velocity = None

    def avoid(
        self, initial_velocity: np.ndarray, limit_velocity_magnitude: bool = True
    ) -> None:
        """Modulate velocity and return DS."""
        # For each control_points
        # -> get distance (minus radius)
        # -> get closest points
        # -> get_weights => how?
        ref_norm = LA.norm(self.reference_direction)

        if not ref_norm:
            # Not modulated when far away from everywhere / in between two obstacles
            return initial_velocity

        if self.normal_direction is None:
            decomposition_matrix = get_orthogonal_basis(
                self.reference_direction / ref_norm, normalize=False
            )

            inv_decomposition = decomposition_matrix.T
        else:
            decomposition_matrix = get_orthogonal_basis(
                self.normal_direction, normalize=False
            )

            decomposition_matrix[:, 0] = self.reference_direction / ref_norm

            inv_decomposition = LA.pinv(decomposition_matrix)

        if self.relative_velocity is not None:
            initial_velocity = initial_velocity - self.relative_velocity
            # breakpoint()

        if not LA.norm(initial_velocity):
            # Trivial velocity modulation
            if self.relative_velocity is None:
                return initial_velocity
            else:
                return initial_velocity - self.relative_velocity

        stretching_matrix = self.stretching_matrix.get(
            ref_norm, self.reference_direction, self.normal_direction, initial_velocity
        )

        modulated_velocity = inv_decomposition @ initial_velocity
        modulated_velocity = stretching_matrix @ modulated_velocity
        modulated_velocity = decomposition_matrix @ modulated_velocity

        # TODO: limit velocity with respect to maximum velocity
        if self.relative_velocity is not None:
            initial_velocity = initial_velocity + self.relative_velocity

        if limit_velocity_magnitude:
            mod_norm = LA.norm(modulated_velocity)
            init_norm = LA.norm(initial_velocity)

            if mod_norm > init_norm:
                modulated_velocity = modulated_velocity * (init_norm / mod_norm)
        
        return modulated_velocity

    def get_weight_from_distances(
        self, distances: np.ndarray,
        directions: np.ndarray = None,
        initial_velocity: np.ndarray = None,
        ):
        """ Returns an array of weights with the same dimensions as distances input. """
        # => get weighted evaluation along the robot
        # to obtain linear + angular velocity
        if any(distances < self.margin_weight):
            warnings.warn("Treat the small-weight case.")

            distances = distances - np.min(distances) + self.margin_weight

        num_points = distances.shape[0]
        weight = (1 / distances) ** self.weight_power * (self.weight_factor / num_points)

        self.distance_weight_sum = np.sum(weight)

        # Reduce wake effect behind an obstacle
        if directions is not None and initial_velocity is not None:
            # TODO: the way the weight is caluclated has to be changed slightly,
            # it needs to be done each 'avoid' funtion to incoorporate this...
            breakpoint()
            dir_weight = (
                1 + np.sum(directions, np.tile(initial_velocity, (1, directions.shape[1]), axis=0))
                / (LA.norm(directions, axis=0)*LA.norm(initial_velocity))
            )
            
            weight = weight * dir_weight

        if (
            self.weight_max_norm is not None
            and self.distance_weight_sum > self.weight_max_norm
        ):
            self.distance_weight_sum = self.weight_max_norm

        if self.distance_weight_sum > 1:
            return weight / self.distance_weight_sum
        else:
            return weight


    def limit_velocity(self):
        raise NotImplementedError()

    def limit_acceleration(self):
        raise NotImplementedError()
