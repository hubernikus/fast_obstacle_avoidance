"""
Utils to simplify the evaluation of laserscan.
"""
# Author: Lukas Huber
# Created: 2021-12-24
# Email: lukas.huber@epfl.ch

import rosbag

import numpy as np

# from numpy import linalg as LA


def reset_laserscan(allscan, position, angle_gap=np.pi / 2):
    # Find the largest gap
    # -> if it's bigger than 90 degrees -> you're outside and should switch
    scangle = np.arctan2(allscan[1, :], allscan[0, :])

    ind_sortvals = np.argsort(scangle)
    sort_vals = scangle[ind_sortvals]
    d_angle = sort_vals - np.roll(sort_vals, shift=1)

    ind_max = np.argmax(d_angle)

    if ind_max > angle_gap:
        flip_low = ind_sortvals[ind_max]
        flip_high = ind_sortvals[ind_max + 1]

        # Reset the reference toa copy before assigning, since it's a mutable object
        allscan = np.copy(allscan)
        allscan[:, flip_low:flip_high] = np.flip(allscan[:, flip_low:flip_high], axis=1)

    return allscan


def import_first_scans(
    robot,
    bag_name="2021-12-13-18-33-06.bag",
    bag_dir="/home/lukas/Code/data_qolo/",
    start_time=None,
):

    # bag_name = '2021-12-13-18-32-13.bag'
    # bag_name = '2021-12-13-18-32-42.bag'
    # bag_name = '2021-12-13-18-33-06.bag'
    # bag_name = '2021-12-13-18-32-13.bag'

    rosbag_name = bag_dir + bag_name

    # my_bag = bagpy.bagreader(rosbag)
    # my_bag = bagpy.bagreader(rosbag_name)
    my_bag = rosbag.Bag(rosbag_name)

    frontscan = None
    rearscan = None

    # for tt in my_bag.topics:
    for topic, msg, t in my_bag.read_messages(
        topics=["/front_lidar/scan", "/rear_lidar/scan"]
    ):
        if start_time is not None and t.to_sec() < start_time:
            continue

        if topic == "/front_lidar/scan" or topic == "/rear_lidar/scan":
            robot.set_laserscan(msg, topic_name=topic)

        if len(robot.laser_data) == len(robot.laser_poses):
            # Make sure go one per element
            break
