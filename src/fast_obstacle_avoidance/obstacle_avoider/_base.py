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


class SingleModulationAvoider:
    def __init__(self):
        self.reference_direction = None
        self.normal_direction = None

        self.norm_power = 3

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
            # Not modulated when far away from everywhere
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

        stretching_matrix = self.get_stretching_matrix(
            ref_norm, self.reference_direction, initial_velocity
        )

        modulated_velocity = inv_decomposition @ initial_velocity
        modulated_velocity = stretching_matrix @ modulated_velocity
        modulated_velocity = decomposition_matrix @ modulated_velocity

        if limit_velocity_magnitude:
            mod_norm = LA.norm(modulated_velocity)
            init_norm = LA.norm(initial_velocity)
            # breakpoint()
            if mod_norm > init_norm:
                modulated_velocity = modulated_velocity * (init_norm / mod_norm)

        return modulated_velocity

    def get_stretching_matrix(
        self,
        ref_norm: float,
        normal_direction: np.ndarray = None,
        initial_velocity: np.ndarray = None,
        free_tail_flow: bool = True,
    ) -> np.ndarray:
        """Get the diagonal stretching matix which plays the main part in the modulation."""
        weight = self.get_weight_from_norm(ref_norm)

        dot_prod = np.dot(normal_direction, initial_velocity)
        if free_tail_flow and dot_prod > 0:

            dot_prod = dot_prod / (
                LA.norm(normal_direction) * LA.norm(initial_velocity)
            )

            # No tail-effect
            normal_stretch = 1

            stretching_vector = np.hstack(
                (1 + (1 - dot_prod) * weight * np.ones(normal_direction.shape[0]))
            )
        else:

            stretching_vector = np.hstack(
                (1 - weight, 1 + weight * np.ones(normal_direction.shape[0] - 1))
            )

        return np.diag(stretching_vector)

    def get_weight_from_norm(self, norm):
        return norm ** (1.0 / self.norm_power)

    def get_weight_from_distances(
        self, distances, weight_factor=3, weight_power=2.0, margin_weight=1e-3
    ):
        # => get weighted evaluation along the robot
        # to obtain linear + angular velocity
        if any(distances < margin_weight):
            warnings.warn("Treat the small-weight case.")

            distances = distances - np.min(distances) + margin_weight

        num_points = distances.shape[0]
        weight = (1 / distances) ** weight_power * (weight_factor / num_points)

        weight_sum = np.sum(weight)

        if weight_sum > 1:
            return weight / weight_sum
        else:
            return weight

    def update_normal_direction(self, ref_dirs, norm_dirs, weights) -> np.ndarray:
        """Update the normal direction of an obstacle."""
        if self.obstacle_environment.dimension == 2:
            norm_angles = np.cross(ref_dirs, norm_dirs, axisa=0, axisb=0)

            norm_angle = np.sum(norm_angles * weights)
            norm_angle = np.arcsin(norm_angle)

            # Add angle to reference direction
            unit_ref_dir = self.reference_direction / LA.norm(self.reference_direction)
            norm_angle += np.arctan2(unit_ref_dir[1], unit_ref_dir[0])

            self.normal_direction = np.array([np.cos(norm_angle), np.sin(norm_angle)])

        elif self.obstacle_environment.dimension == 3:
            norm_angles = np.cross(norm_dirs, ref_dirs, axisa=0, axisb=0)

            norm_angle = np.sum(
                norm_angles * np.tile(weights, (relative_position.shape[0], 1)), axis=1
            )
            norm_angle_mag = LA.norm(norm_angles)
            if not norm_angle_mag:  # Zero value
                self.normal_direction = copy.deepcopy(self.reference_direction)

            else:
                norm_rot = Rotation.from_vec(
                    self.normal_direction / norm_angle_mag * np.arcsin(norm_angle_mag)
                )

                unit_ref_dir = self.reference_direction / norm_ref_dir

                self.normal_direction = norm_rot.apply(unit_ref_dir)

        else:
            raise NotImplementedError(
                "For higher dimensions it is currently not defined."
            )

        return self.normal_direction
