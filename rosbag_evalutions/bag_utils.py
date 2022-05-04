""" Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2022-05-03
# Github: hubernikus

import os
from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
from numpy import linalg as LA
import pandas as pd

# import rosbag_pandas
# from rosbags.dataframe import get_dataframe
# from rosbags.highlevel import AnyReader
from rosbags.rosbag1 import Reader
from rosbags.serde import deserialize_cdr, ros1_to_cdr

import rosbag

from vartools.states import ObjectPose
from fast_obstacle_avoidance.control_robot import QoloRobot

import seaborn as sns

# Set logger
# logger = logging.getLogger("personal")
# formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)-12s %(levelname)-8s %(message)s",
    datefmt="%m-%d %H:%M",
)

## Functions
def print_topics(bagname=None, bag_dir=None):
    if bag_dir is None:
        bag_dir = "../../data_qolo/marketplace_lausanne_2022_01_28"
    if bagname is None:
        bagname = os.path.join(bag_dir, "2022-01-28-13-20-46.bag")

    logging.info("Printing topics")

    # Print datatypes
    with Reader(bagname) as reader:
        for connection in reader.connections:
            print(connection.topic, connection.msgtype)


@dataclass
class Recording:
    position_x: np.ndarray
    position_y: np.ndarray
    orientation: np.ndarray

    velocity_x: np.ndarray
    velocity_y: np.ndarray
    velocity_angular: np.ndarray


def get_bagvalues(bagname):
    # Asign values
    pose_x = []
    pose_y = []
    pose_theta = []

    vel_x = []
    vel_y = []
    vel_angular = []

    with Reader(bagname) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == "/qolo/pose2D":
                msg = deserialize_cdr(
                    ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                )
                pose_x.append(msg.x)
                pose_y.append(msg.y)
                pose_theta.append(msg.theta)

            if connection.topic == "/qolo/twist":
                msg = deserialize_cdr(
                    ros1_to_cdr(rawdata, connection.msgtype), connection.msgtype
                )
                vel_x.append(msg.twist.linear.x)
                vel_y.append(msg.twist.linear.y)
                vel_angular.append(msg.twist.angular.z)

    recording = Recording(
        position_x=np.array(pose_x),
        position_y=np.array(pose_y),
        orientation=np.array(pose_theta),
        velocity_x=np.array(vel_x),
        velocity_y=np.array(vel_y),
        velocity_angular=np.array(vel_y),
    )

    return recording


if (__name__) == "__main__":
    # Bag tuples
    bag_dir = "../data_qolo/marketplace_lausanne_2022_01_28/"

    bag_tuples = [
        # Does one loop (to far away), slow velocity ~ 0.4, important is the start
        ("2022-01-28-13-50-27.bag", "Sampled slow"),
        # Goes back and forwards once
        ("2022-01-28-13-33-32.bag", "Sampled"),
        ("2022-01-28-13-20-25.bag", "Sampled"),  # Very short -> nonusefull
        ("2022-01-28-13-20-46.bag", "Sampled"),
        ("2022-01-28-14-00-39.bag", "Sampled"),
        ("2022-01-28-14-03-04.bag", "Sampled"),
        ("2022-01-28-14-08-47.bag", "Sampled"),
    ]
    bag_names = [name for name, _ in bag_tuples]
    experiment_id = [name[-9:-7] + name[-6:-4] for name, _ in bag_tuples]
    detection_type = [dettype for _, dettype in bag_tuples]

    # bag_name = os.path.join(bag_dir, "2022-01-28-13-20-46.bag")
    # bag_name = os.path.join(bag_dir, bag_tuples[1][0])

    if False:
        logging.info("Default importing in progress...")

        ## Values
        logging.info("Start data-importing.")
        recording_list = []
        for ii, bagname in enumerate(bag_names):
            bagname_dir = os.path.join(bag_dir, bagname)

            recording_list.append(get_bagvalues(bagname_dir))

        logging.info("Done importing.")

    ## Analyse file
    # reader = open(Reader(bag_name))  # WARNING: not good practice but easy datascience
    # main_reader = Reader(bag_name)
    # reader = main_reader.__enter__()

    # Close only now
    # main_reader.__exit__(None, None, None)

    ## Get topics
    # rawdata is of type bytes and contains serialized message
    # msg = deserialize_cdr(rawdata, "geometry_msgs/msg/Quaternion")

    ## Done
