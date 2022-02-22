""" Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2021-12-27
# Email: lukas.huber@epfl.ch

import sys
import os

import datetime
import subprocess

from timeit import default_timer as timer

import math

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

# import pandas as pd
import rosbag

from vartools.states import ObjectPose
from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from fast_obstacle_avoidance.control_robot import QoloRobot
from fast_obstacle_avoidance.utils import laserscan_to_numpy
from fast_obstacle_avoidance.obstacle_avoider import FastLidarAvoider
from fast_obstacle_avoidance.laserscan_utils import import_first_scans

# def multicolor_line():


class MultiPloter:
    def rosbag_generator(self, my_bag, bag_dir=None, bag_name=None, dx_max=1):
        """Generator member function which allows state-storing and updating.
        Yields/Return 0 for success; all data is stored in `self`."""
        if my_bag is None:
            rosbag_name = bag_dir + bag_name
            # my_bag = bagpy.bagreader(rosbag)
            # my_bag = bagpy.bagreader(rosbag_name)
            my_bag = rosbag.Bag(rosbag_name)

        for topic, msg, t in my_bag.read_messages(
            topics=[
                # "/front_lidar/scan",
                # "/rear_lidar/scan",
                # "/rwth_tracker/tracked_persons",
                "/qolo/user_commands",  # -> input
                "/qolo/remote_commands",  # -> output
                # "/qolo/twist", # (? and this?)
                # "/qolo/pose2D",
            ]
        ):

            self.ros_time = t.to_sec()

            if topic == "/front_lidar/scan" or topic == "/rear_lidar/scan":
                continue
                # self.robot.set_laserscan(msg, topic_name=topic, save_intensity=True)

            if topic == "/rwth_tracker/tracked_persons":
                breakpoint()
                # msg_persons = msg

            if topic == "/qolo/pose2D":
                continue
                # msg_qolo = msg
                # self.robot.pose.position = np.array([msg.x, msg.y])
                # self.robot.pose.orientation = msg.theta

            # Input from user via remote / belt
            if topic == "/qolo/user_commands":
                self.RemoteJacobian = np.diag([1, 0.15])
                self.initial_velocity = LA.inv(np.diag([1, 0.15])) @ np.array(
                    [msg.data[1], msg.data[2]]
                )

                self.initial_velocity = (
                    self.robot.rotation_matrix.T @ self.initial_velocity
                )

            # Output to qolo wheels
            if topic == "/qolo/remote_commands":
                # self.Jacobian = np.diag([1,  0.0625])
                self.modulated_velocity = LA.inv(np.diag([1, 0.0625])) @ np.array(
                    [msg.data[1], msg.data[2]]
                )
                self.modulated_velocity = (
                    self.robot.rotation_matrix.T @ self.modulated_velocity
                )

                # Caluclate Deviation
                if LA.norm(self.initial_velocity) and LA.norm(self.modulated_velocity):
                    initial_velocity = self.initial_velocity / LA.norm(
                        self.initial_velocity
                    )
                    modulated_velocity = self.modulated_velocity / LA.norm(
                        self.modulated_velocity
                    )
                    dot_prod = np.dot(initial_velocity, modulated_velocity)

                    # Make sure no numerical errors occur
                    dot_prod = np.maximum(dot_prod, -1)
                    dot_prod = np.minimum(dot_prod, 1)

                    angle = np.arccos(dot_prod)

                    angle_dir = np.cross(initial_velocity, modulated_velocity)
                    angle = np.copysign(angle, angle_dir)

                    if np.isnan(angle):
                        breakpoint()

                    self.angle_list.append(angle)

            yield 0

        # All done - no iteration possible anymore
        yield 1

    def __init__(self, robot, my_bag):
        self.dimension = 2

        self.robot = robot

        self.angle_list = []
        self.initial_velocity = np.zeros(self.dimension)
        self.modulated_velocity = np.zeros(self.dimension)

        # Initialization of generator and do one step
        self.my_generator = self.rosbag_generator(my_bag)
        self.ros_state = next(self.my_generator)

    def create(self, save_figure=False, bag_name=None, num_bars=50):
        delta_angle = 2 * np.pi / num_bars

        # Iterate through all of them(!)
        for ii in self.my_generator:
            continue

        # Create plot
        self.fig = plt.figure()
        self.ax = plt.subplot(111, polar=True)

        min_val = -np.pi
        max_val = np.pi
        range_val = max_val - min_val

        # Make sure to odd number of bins / even number of bin-spaces
        if num_bars % 2:  # Is odd
            num_bars = num_bars + 1

        angle_bins = np.linspace(min_val, max_val, num_bars)
        # hist_vals = np.histogram(self.angle_list, bins=bins=angle_bins)

        if not len(self.angle_list):
            return

        bars = self.ax.hist(
            self.angle_list,
            bins=angle_bins,
            # edgecolor="white",
            density=True,
            # color='',
        )

        # bars[0] = bars[0] / np.sum(bars[0])

        # Reduce argmax to 'second-largest' value
        max_val_factor = 1.1
        args_sorted = np.argsort(bars[0])
        max_val = bars[0][args_sorted[-2]] * max_val_factor

        from matplotlib.ticker import PercentFormatter

        # self.ax.set_major_formatter(PercentFormatter(1))

        # Y ticks + range
        perc_factor = 100 / np.sum(bars[0])
        num_y_ticks = 5

        if not perc_factor:
            return

        max_val = (
            round(max_val * perc_factor / ((num_y_ticks - 1)))
            * (num_y_ticks - 1)
            / perc_factor
        )

        self.ax.set_ylim([0, max_val])
        y_ticks = np.linspace(0, max_val, num_y_ticks)
        # self.ax.set_yticks(y_ticks / np.sum(bars[0])*100)

        self.ax.set_yticks(y_ticks)
        y_labels = [f"{round(iy*perc_factor)}%" for iy in y_ticks]
        y_labels[0] = ""
        self.ax.set_yticklabels(y_labels)
        # self.ax.set_yscale('log')

        self.ax.set_theta_zero_location("N")
        self.ax.set_rlabel_position(300)
        # self.ax.set_theta_direction(-1)
        # self.ax.set_rscale('log')
        # self.ax.invert_yaxis()

        num_x_ticks = 8
        x_ticks = np.linspace(0, 2 * np.pi, num_x_ticks, endpoint=False)
        self.ax.set_xticks(x_ticks)

        x_ticks = np.copy(x_ticks)
        for ii, x_val in enumerate(x_ticks):
            if x_val > np.pi:
                x_ticks[ii] = x_val - 2 * np.pi

        self.ax.set_xticklabels(
            [f"{round(ix/np.pi*180)}\N{DEGREE SIGN}" for ix in x_ticks]
        )
        # self.ax.set_xticklabels([f"{round(ix/np.pi*180}deg" for iy in y_ticks])
        # self.ax.set_xticklabels([f"{round(ix/np.sum(bars[0])*100)}%" for ix in x_ticks])

        # self.ax.grid(zorder=-1)
        self.ax.set_axisbelow(True)

        # self.ax.set_ylabel([f"{iy/np.sum(bars[0])*100}%" for iy in y_ticks])

        # plt.xticks(angle_bins)

        # angles = np.linspace(min_val, max_val, num_bars + 1)
        # angles = (angles[1:] + angles[:-1]) * 0.5

        if save_figure:
            figure_name = "bag_moduation_angle_"
            if bag_name is not None:
                figure_name = figure_name + bag_name[:-4]

            plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def bag_evaluate_deviation(my_bag, bag_name, save_figure=True):
    qolo = QoloRobot(
        pose=ObjectPose(position=np.array([0, 0]), orientation=0 * np.pi / 180)
    )

    my_ploter = MultiPloter(
        robot=qolo,
        my_bag=my_bag,
    )

    my_ploter.create(save_figure=save_figure, bag_name=bag_name)

    # my_ploter.ax.set_xlim([-7.4, 2.4])
    # my_ploter.ax.set_ylim([-4, 4])


def main_batch_processing(bag_dir):
    """This script is for batch-processing of the ros-bag recording with the qolo."""
    # Use aspect ratio of 16:9 [youtube standard ration] for x_lim / y_lim
    # bag_list = glob.glob(bag_dir + "/*.bag")
    bag_list = os.listdir(bag_dir)

    for bag_name in bag_list:
        print(f"Trying bag {bag_name}")
        my_bag = rosbag.Bag(bag_dir + bag_name)

        bag_evaluate_deviation(my_bag, bag_name)


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    bag_dir = "../data_qolo/marketplace_lausanne_2022/"
    main_batch_processing(bag_dir)

    # Single bags
    single_bag = False
    if single_bag:
        # Fast ones
        # bag_dir = "../data_qolo/indoor_with_david_2022_01/"
        # bag_name = "2022-01-26-17-50-23.bag"

        bag_name = "2022-01-28-13-33-32.bag"
        # bag_name = "2022-01-28-14-00-39.bag"

        import_again = True
        if import_again or not "my_bag" in locals():
            my_bag = rosbag.Bag(bag_dir + bag_name)

        main(my_bag, bag_name=bag_name)
