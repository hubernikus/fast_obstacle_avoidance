import sys 
import os

import numpy as np
import matplotlib.pyplot as plt

# import pandas as pd

import rosbag

# import rosbag
from vartools.states import ObjectPose

from fast_obstacle_avoidance.control_robot import ControlRobot
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider


def laserscan_to_numpy(msg, dimension=2, delta_angle=0, delta_position=None) -> np.ndarray:
    num_points = len(msg.ranges)

    ranges = np.array(msg.ranges)
    ind_real = np.isfinite(ranges)

    ranges = ranges[ind_real]
    angles = np.arange(num_points)[ind_real]*msg.angle_increment + (msg.angle_min + delta_angle)
    positions = np.tile(ranges, (dimension, 1)) * np.vstack((np.cos(angles), np.sin(angles)))

    if delta_position is not None:
        # Rotate
        cos_val = np.cos(delta_angle)
        sin_val = np.sin(delta_angle)
        rot_matr = np.array([[cos_val, sin_val],
                             [-sin_val, cos_val]])
        delta_position = rot_matr @ delta_position

        postitions = positions + np.tile(delta_position, (positions.shape[1], 1)).T
    
    return positions

def get_topics(rosbag_name):
    import bagpy
    # get the list of topics
    print(bagpy.bagreader(rosbag_name).topic_table)
    

def main():
    bag_dir = '/home/lukas/Code/data_qolo/'
    # bag_name = '2021-12-13-18-32-13.bag'
    # bag_name = '2021-12-13-18-32-42.bag'
    bag_name = '2021-12-13-18-33-06.bag'
    # bag_name = '2021-12-13-18-32-13.bag'
    rosbag_name = bag_dir + bag_name
    
    # my_bag = bagpy.bagreader(rosbag)
    # my_bag = bagpy.bagreader(rosbag_name)
    my_bag = rosbag.Bag(rosbag_name)

    

    frontscan = None
    rearscan = None
    
    # for tt in my_bag.topics:
    for topic, msg, t in my_bag.read_messages(topics=[
        '/front_lidar/scan',
        '/rear_lidar/scan'
        ]):
        if topic == '/front_lidar/scan':
            frontscan = laserscan_to_numpy(msg)
            
        elif topic == '/rear_lidar/scan':
            rearscan = laserscan_to_numpy(msg, delta_angle=np.pi, delta_position=[0.5, 0])

            
        if frontscan is not None and rearscan is not None:
            break

    allscan = np.hstack((rearscan, frontscan))

    qolo = ControlRobot(
        control_points=np.array([[0, 0],
                                 # [0.5, 0],
                                 ]).T,
        
        control_radiuses=np.array([0.5,
                                   # 0.4,
                                   ]),
        pose = ObjectPose(position=[0, 0], orientation=30*np.pi/180)
    )

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(0, 0, 'o', color='k')
    ax.plot(frontscan[0, :], frontscan[1, :], '.', color='r')
    ax.plot(rearscan[0, :], rearscan[1, :], '.', color='b')

    ax.set_xlim([-10, 5])
    ax.set_ylim([-5, 5])
    ax.set_aspect('equal')

    qolo.plot2D(ax)
    # print(tt)
    ax.grid()
    # for topic, msg, t in my_bag.read_messages(topics=[
         # '/front_lidar/scan',
         # '/rear_lidar/scan'
    # ]):
        # breakpoint()

    
    

if (__name__) == "__main__":
    plt.close('all')
    main()
    pass
