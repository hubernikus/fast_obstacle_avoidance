"""
Simplified obstacle avoidance for mobile for mobile robots based on
DS & reference_direction

The algorithm is developed for shared control,
> human will know the best path, just be fast and don't get stuck
(make unexpected mistakes)
"""

import numpy as np
import matplotlib.pyplot as plt

from fast_obstacle_avoidance.control_robot import ControlRobot


class FastObstacleAvoider:
    """ To proof:
    -> reference direction becomes normal (!) when getting very close (!)
    -> obstacle is being avoided when very close(!) [reference -> normal]
    -> No local minima (and maybe even convergence to attractor)
    -> add slight repulsion along the normal direction (when getting closer)
    """
    def __init__(self, robot: ControlRobot):
        self.robot = robot

    def get_velocity(self, initial_velocity: np.ndarray, laser_scan: np.ndarray):
        # For each control_points
        # -> get distance (minus radius)
        # -> get closest points
        # -> get_weights => how?

        # => get weighted evaluation along the robot
        # to obtain linear + angular velocity
        
        pass


    def modulate(self, initial_velocity, reference_directions):
        # For all reference directions
        pass

    def limit_acceleration(self):
        pass

    def get_weights(self, distances: np.ndarray) -> np.ndarray:
        """ Returns weight with the same shape as distances. """
        return distances
    
    def limit_velocity(self):
        pass

