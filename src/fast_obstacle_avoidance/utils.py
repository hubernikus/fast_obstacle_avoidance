"""
Various utils used for the fast-obstacle-avoidance
"""
import numpy as np
from numpy import linalg as LA

from scipy.spatial.transform import Rotation as R


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


def depreciated(*args, **kwargs):
    # def obstacle_list_in_local_frame(msg, robot):
    breakpoint()

    tt = gmsg.TransformStamped()

    # Prepare broadcast message
    # Copy in pose values to transform
    # tt.transform.translation = pose.position
    position_qolo = np.array([qolo_pose.x, qolo_pose.y])
    orientation_qolo = np.array([qolo_pose.theta])

    cos_, sin_ = np.cos(ori), np.sin(ori)
    orientatin_matrix = np.array([[cos_, sin_], [sin_, cos_]])

    # tt.transform.translation.x = qolo_pose.x
    # tt.transform.translation.y = qolo_pose.y
    # quat = Rotation.from_quat([qolo_pose.theta, 0, 0], 'zyx').as_quat()
    # tt.transform.rotation = quat

    obs_list = []
    for track in range(msg.tracks):
        # pose_local = tf2.do_transform_pose(tt, track.pose)

        pose_local = (tt, track.pose)
        euler = Rotation.from_quat(
            [
                pose_local.pose.quaternion.x,
                pose_local.pose.quaternion.y,
                pose_local.pose.quaternion.z,
                pose_local.pose.quaternion.w,
            ]
        ).as_euler("zyx")

    return obs_list
