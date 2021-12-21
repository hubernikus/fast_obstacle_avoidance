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

from fast_obstacle_avoidance.control_robot import ControlRobot

from vartools.linalg import get_orthogonal_basis


class SingleModulationAvoider:
    def __init__(self):
        self.reference_direction = None
        self.normal_direction = None
        
    def avoid(self,
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

        if self.normal_direction is None:
            decomposition_matrix = get_orthogonal_basis(
                self.reference_direction/ref_norm, normalize=False)

            inv_decomposition = decomposition_matrix.T
        else:
            decomposition_matrix = get_orthogonal_basis(
                self.normal_direction, normalize=False)
            
            decomposition_matrix[:, 0] = self.reference_direction/ref_norm

            inv_decomposition = LA.pinv(decomposition_matrix)

        stretching_matrix = self.get_stretching_matrix(
            ref_norm, self.reference_direction, initial_velocity)

        # breakpoint()
        modulated_velocity = inv_decomposition @ initial_velocity
        modulated_velocity = stretching_matrix @ modulated_velocity
        modulated_velocity = decomposition_matrix @ modulated_velocity

        if limit_velocity_magnitude:
            mod_norm = LA.norm(modulated_velocity)
            init_norm = LA.norm(initial_velocity)
            
            if mod_norm > init_norm:
                modulated_velocity = modulated_velocity * (init_norm/mod_norm)
            
        return modulated_velocity

    def get_stretching_matrix(
        self, ref_norm: float,
        normal_direction: np.ndarray = None,
        initial_velocity: np.ndarray = None,
        free_tail_flow: bool = True) -> np.ndarray:
        """ Get the diagonal stretching matix which plays the main part in the modulation."""
        weight = self.get_weight_from_norm(ref_norm)

        if free_tail_flow and np.dot(normal_direction, initial_velocity) > 0:
            # No tail-effect
            normal_stretch = 1
        else:
            
            normal_stretch = 1 - weight
        
        stretching_vector = np.hstack((
            normal_stretch,
            1+weight * np.ones(normal_direction.shape[0]-1)
            ))
        
        return np.diag(stretching_vector)

    def get_weight_from_norm(self, norm):
        return norm

    def get_weight_from_distances(
        self, distances, weight_factor=3, weight_power=2.0, margin_weight=1e-3):
        # => get weighted evaluation along the robot
        # to obtain linear + angular velocity
        if any(distances < margin_weight):
            warnings.warn("Treat the small-weight case.")

            distances = distances - np.min(distances) + margin_weight

        num_points = distances.shape[0]
        weight = (1 / distances)**weight_power * (weight_factor/num_points)

        weight_sum = np.sum(weight)
        
        if weight_sum > 1:
            return weight / weight_sum
        else:
            return weight

    def update_normal_direction(self, ref_dirs, norm_dirs, weights) -> np.ndarray:
        """ Update the normal direction of an obstacle. """
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
                norm_angles * np.tile(weights, (relative_position.shape[0], 1)),
                axis=1
            )
            norm_angle_mag = LA.norm(norm_angles)
            if not norm_angle_mag: # Zero value
                self.normal_direction = copy.deepcopy(self.reference_direction)
                
            else:
                norm_rot = Rotation.from_vec(
                    self.normal_direction / norm_angle_mag * np.arcsin(norm_angle_mag)
                )

                unit_ref_dir = self.reference_direction / norm_ref_dir
                
                self.normal_direction = norm_rot.apply(unit_ref_dir)

        else:
            raise NotImplementedError(
                "For higher dimensions it is currently not defined.")

        return self.normal_direction

    

class FastObstacleAvoider(SingleModulationAvoider):
    def __init__(self, obstacle_environment):
        """ Initialize with obstacle list
        """
        self.obstacle_environment = obstacle_environment

        super().__init__()

        if (self.obstacle_environment.dimension ==3):
            from scipy.spatial.transform import Rotation as R

            
    def update_reference_direction(self, position):
        norm_dirs = np.zeros((self.obstacle_environment.dimension,
                             self.obstacle_environment.n_obstacles))
        ref_dirs = np.zeros(norm_dirs.shape)
        relative_distances = np.zeros((norm_dirs.shape[1]))

        for it, obs in enumerate(self.obstacle_environment):
            norm_dirs[:, it] = obs.get_normal_direction(
                position, in_global_frame=True)
            
            ref_dirs[:, it] = (-1)*obs.get_reference_direction(
                position, in_global_frame=True)

            relative_distances[it] = obs.get_gamma(
                position, in_global_frame=True)-1

        weights = self.get_weight_from_distances(relative_distances)
        
        self.reference_direction = np.sum(
            ref_dirs * np.tile(weights, (ref_dirs.shape[0], 1)),
            axis=1
            )

        norm_ref_dir = LA.norm(self.reference_direction)
        if not norm_ref_dir:
            self.normal_direction = np.zeros(reference_direction.shape)
            return

        self.normal_direction = self.update_normal_direction(
            ref_dirs, norm_dirs, weights
            )
    

class FastLidarAvoider(SingleModulationAvoider):
    """ To proof:
    -> reference direction becomes normal (!) when getting very close (!)
    -> obstacle is being avoided when very close(!) [reference -> normal]
    -> No local minima (and maybe even convergence to attractor)
    -> add slight repulsion along the normal direction (when getting closer)
    """
    def __init__(self, robot: ControlRobot, evaluate_normal: bool = False) -> None:
        self.robot = robot

        self.evaluate_normal = evaluate_normal
        self.max_angle_ref_norm = 80*np.pi/180
        super().__init__()

    def update_laserscan(self, laser_scan: np.ndarray) -> None:
        self.reference_direction = self.get_reference_direction(
            laser_scan)

    def get_reference_direction(self, laser_scan: np.ndarray) -> np.ndarray:
        laser_scan, ref_dirs, relative_distances = (
            self.robot.get_relative_positions_and_dists(laser_scan)
            )
        weights = self.get_weight_from_distances(relative_distances)

        self.reference_direction = (-1)*np.sum(
            ref_dirs * np.tile(weights, (ref_dirs.shape[0], 1)),
            axis=1
            )

        norm_ref_dir = LA.norm(self.reference_direction)
        
        if self.evaluate_normal and norm_ref_dir:
            # Only evaluate in presence of non-trivial reference direction
            if laser_scan.shape[0] != 2:
                raise NotImplementedError("Only done for ii>1")
            
            # Reduce data to necesarry ones only
            ind_nonzero = weights > 0
            weights = weights[ind_nonzero]
            ref_dirs = ref_dirs[:, ind_nonzero]
            laser_scan = laser_scan[:, ind_nonzero]

            tangents = laser_scan - np.roll(laser_scan, shift=1, axis=1)
            
            normals = np.vstack(((-1)*tangents[1, :], tangents[0, :]))
            
            # Remove any which happended through overlap
            # Matrix dot-product
            ind_bad = (np.sum(ref_dirs*normals, axis=0) < 0)

            if any(ind_bad):
                normals[:, ind_bad] = ref_dirs[:, ind_bad]

            normals = normals / np.tile(LA.norm(normals, axis=0), (normals.shape[0], 1))

            # Average over two steps
            normals = (normals + np.roll(normals, shift=1, axis=1)) / 2.0
            normals = normals / np.tile(LA.norm(normals, axis=0), (normals.shape[0], 1))

            norm_angles = np.cross(ref_dirs, normals, axisa=0, axisb=0)
            ind_critical = np.abs(norm_angles) > np.sin(self.max_angle_ref_norm)
            if any(ind_critical):
                norm_angles[ind_critical] = np.copysign(
                    np.sin(self.max_angle_ref_norm), norm_angles[ind_critical]
                    )
            
            norm_angle = np.sum(norm_angles * weights)
            norm_angle = np.arcsin(norm_angle)

            # Add angle to reference direction
            unit_ref_dir = self.reference_direction / norm_ref_dir
            norm_angle += np.arctan2(unit_ref_dir[1], unit_ref_dir[0])
            
            self.normal_direction = np.array([np.cos(norm_angle),
                                              np.sin(norm_angle)])

        # Make sure reference direction points away from center
        print('devi', np.arcsin(np.cross(self.reference_direction,
                                         self.normal_direction))
              )
        return self.reference_direction
            
    def limit_velocity(self):
        raise NotImplementedError()

    def limit_acceleration(self):
        raise NotImplementedError()

