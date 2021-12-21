"""
Various utils used for the fast-obstacle-avoidance
"""
import numpy as np
from numpy import linalg as LA


def laserscan_to_numpy(
    msg, dimension=2, delta_angle=0, delta_position=None, pose=None
) -> np.ndarray:
    """Returns a numpy array of the a ros-laserscan data."""
    if pose is not None:
        delta_angle = pose.orientation
        if LA.norm(pose.position):
            delta_position = pose.position

    num_points = len(msg.ranges)

    ranges = np.array(msg.ranges)
    ind_real = np.isfinite(ranges)

    ranges = ranges[ind_real]
    angles = np.arange(num_points)[ind_real] * msg.angle_increment + (
        msg.angle_min + delta_angle
    )

    positions = np.tile(ranges, (dimension, 1)) * np.vstack(
        (np.cos(angles), np.sin(angles))
    )

    if delta_position is not None:
        # Rotate
        # cos_val = np.cos(delta_angle)
        # sin_val = np.sin(delta_angle)

        # rot_matr = np.array([[cos_val, sin_val],
        # [-sin_val, cos_val]])

        # delta_position = rot_matr @ delta_position
        positions = positions + np.tile(delta_position, (positions.shape[1], 1)).T

    return positions
