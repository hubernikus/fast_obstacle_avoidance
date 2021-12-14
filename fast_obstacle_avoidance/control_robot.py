from dataclasses import dataclass

import numpy as np
from numpy import linalg as LA

from vartools.states import ObjectPose


@dataclass
class ControlRobot:
    control_points: np.ndarray
    control_radiuses: np.ndarray

    pose: ObjectPose = None
    robot_image = None
    
    @property
    def num_control_points(self) -> int:
        return self.control_radiuses.shape[0]

    def get_relative_positions_and_dists(self, positions):
        """ Get normalized (relative) position and (relative) surface distance. """
        rel_pos = positions - np.tile(self.pose.position, (positions.shape[1], 1)).T
        rel_dist = LA.norm(rel_pos, axis=0)
        
        rel_pos = rel_pos / np.tile(rel_dist, (rel_pos.shape[0], 1))
        rel_dist = rel_dist - self.control_radiuses[0]
        
        return rel_pos, rel_dist

    def plot2D(self, ax, num_points=30) -> None:
        if self.robot_image is not None:
            raise NotImplementedError("Nothing is being done so far for robot images.")

        angles = np.linspace(0, 2*np.pi, num_points)
        unit_circle = np.vstack((np.cos(angles), np.sin(angles)))

        for ii in range(self.control_radiuses.shape[0]):
            ctrl_point = self.pose.transform_position_from_local_to_reference(
                self.control_points[:, ii])
            temp_cicle = (unit_circle * self.control_radiuses[ii]
                          + np.tile(ctrl_point, (num_points, 1)).T)

            ax.plot(temp_cicle[0, :], temp_cicle[1, :], '--', color='k')
