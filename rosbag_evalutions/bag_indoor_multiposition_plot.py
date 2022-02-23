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
                "/front_lidar/scan",
                "/rear_lidar/scan",
                "/rwth_tracker/tracked_persons",
                "/qolo/user_commands",  # -> input
                "/qolo/remote_commands",  # -> output
                # "/qolo/twist", # (? and this?)
                "/qolo/pose2D",
            ]
        ):

            self.ros_time = t.to_sec()

            if topic == "/front_lidar/scan" or topic == "/rear_lidar/scan":
                self.robot.set_laserscan(msg, topic_name=topic, save_intensity=True)

            if topic == "/rwth_tracker/tracked_persons":
                breakpoint
                # msg_persons = msg

            if topic == "/qolo/pose2D":
                msg_qolo = msg
                self.robot.pose.position = np.array([msg.x, msg.y])
                self.robot.pose.orientation = msg.theta

                # Add to list
                self.position_list.append(self.robot.pose.position)

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

            yield 0

        # All done - no iteration possible anymore
        yield 1

    def __init__(self, robot, my_bag, width_x, width_y, visual_times=None):
        if visual_times is None:
            self.visual_times = [0, 1]
        else:
            self.visual_times = visual_times
            
        self.dimension = 2

        delta_orientation = 0 / 180.0 * np.pi

        self.initial_velocity = None
        self.modulated_velocity = None
        self.start_time = None

        self.robot = robot

        self.width_x = width_x
        self.width_y = width_y

        self.position_list = []

        # Initialization of variables
        self.my_generator = self.rosbag_generator(my_bag)
        while (
            self.initial_velocity is None
            or self.modulated_velocity is None
        ):
            self.ros_state = next(self.my_generator)
            print("Waiting to recieve")
            
        self.start_time = self.ros_time

    def create(self, save_figure=False, bag_name=None, do_subplots=True,
               time_text=True, draw_laserscan=True):
        global_ctrl_point = np.zeros(self.dimension)

        # self.fig, self.ax = plt.subplots(figsize=(16, 10))
        # Reset position list
        self.position_list = []

        if do_subplots:
            n_cols, n_rows = 3, 2
            # self.fig, self.axs = plt.subplots(figsize=(4, 4))
            # self.fig, self.axs = plt.subplots(n_rows, n_cols, figsize=(7, 5))

        for ii, vtime in enumerate(self.visual_times):
            # self.ax = self.axs[ii % n_rows, int(ii / n_rows)]
            if do_subplots:
                self.ax = self.axs[int(ii / n_cols), ii % n_cols]
            else:
                # self.fig, self.ax = plt.subplots(figsize=(4, 4))
                self.fig, self.ax = plt.subplots(figsize=(8, 8))

            while self.ros_time - self.start_time < vtime and not (self.ros_state):
                self.ros_state = next(self.my_generator)
                continue

            # Draw big marker
            # self.ax.plot(self.robot.pose.position[0],
            # self.robot.pose.position[1],
            # 'ko')

            max_vel = 1.0
            if LA.norm(self.initial_velocity) > max_vel:
                self.initial_velocity = (
                    self.initial_velocity / LA.norm(self.initial_velocity) * max_vel
                )

            if LA.norm(self.modulated_velocity) > max_vel:
                self.modulated_velocity = (
                    self.modulated_velocity / LA.norm(self.modulated_velocity) * max_vel
                )

            # Draw velocity arrows
            arrow_scale = 0.5
            arrow_width = 0.07
            arrow_headwith = 0.4
            margin_velocity_plot = 1e-3
            self.ax.arrow(
                self.robot.pose.position[0] + global_ctrl_point[0],
                self.robot.pose.position[1] + global_ctrl_point[1],
                arrow_scale * self.initial_velocity[0],
                arrow_scale * self.initial_velocity[1],
                width=arrow_width,
                head_width=arrow_headwith,
                # color="g",
                color="#008080",
                label="Initial chommand",
            )

            self.ax.arrow(
                self.robot.pose.position[0] + global_ctrl_point[0],
                self.robot.pose.position[1] + global_ctrl_point[1],
                arrow_scale * self.modulated_velocity[0],
                arrow_scale * self.modulated_velocity[1],
                width=arrow_width,
                head_width=arrow_headwith,
                # color="b",
                # color='#213970',
                color="#000080",
                label="Modulated velocity",
            )

            # if ii == 0:
            # self.ax.legend(loc="upper left")

            # if vtime == 112.3:
            # Chose which time to plot the laserscan

            if True:
                intensities = self.robot.get_all_intensities()

                laserscan = self.robot.get_allscan(in_robot_frame=False)

            if draw_laserscan:
                self.ax.scatter(
                    laserscan[0, :],
                    laserscan[1, :],
                    # c='black',
                    c=intensities,
                    cmap="copper",
                    # cmap='hot',
                    # ".",
                    # color=self.obstacle_color,
                    s=7.0,
                    # alpha=(intensities/255.),
                    # alpha=(1-intensities/255.),
                    alpha=0.8,
                    zorder=-1,
                )

            # self.position_list = np.array(self.position_list).T
            # self.ax.plot(
            # self.position_list[0, :],
            # self.position_list[1, :],
            # '--',
            # color='black',
            # )

            # Just define both if one is not defined
            sensor_center = np.mean(laserscan, axis=1)

            self.ax.set_aspect("equal")
            # self.ax.grid(zorder=-3.0)

            self.x_lim = [
                sensor_center[0] - self.width_x / 2,
                sensor_center[0] + self.width_x / 2,
            ]

            self.y_lim = [
                sensor_center[1] - self.width_y / 2,
                sensor_center[1] + self.width_y / 2,
            ]

            self.ax.set_xlim(self.x_lim)
            self.ax.set_ylim(self.y_lim)

            self.ax.tick_params(
                axis="both",
                which="major",
                labelbottom=False,
                labelleft=False,
                bottom=False,
                top=False,
                left=False,
                right=False,
            )

            self.robot.plot_robot(self.ax)

            if time_text:
                self.ax.text(
                    self.x_lim[1] - 0.22 * (self.x_lim[1] - self.x_lim[0]),
                    self.y_lim[1] - 0.08 * (self.y_lim[1] - self.y_lim[0]),
                    f"{str(round(vtime - self.visual_times[0], 1))} s",
                    fontsize=10,
                    backgroundcolor="#FFFFFF",
                )

            if not do_subplots and save_figure:
                figure_name = "bag_snipplet_"
                if bag_name is not None:
                    figure_name = figure_name + bag_name[:-4]

                figure_name = figure_name + f"_fig_{str(ii)}"

                plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

        # create some space below the plots by increasing the bottom-value

        if not do_subplots:
            return
        
        self.fig.tight_layout()
        self.fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.09)
        self.axs.flatten()[-2].legend(
            loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3
        )

        if save_figure:
            figure_name = "bag_snipplet_"
            if bag_name is not None:
                figure_name = figure_name + bag_name[:-4]

            plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def main(my_bag, bag_name):
    qolo = QoloRobot(
        pose=ObjectPose(position=np.array([0, 0]), orientation=0 * np.pi / 180)
    )

    my_ploter = MultiPloter(
        robot=qolo,
        my_bag=my_bag,
        width_x=8,
        width_y=8,
        visual_times=[
            100.2,
            104.4,
            112.3,
            114.2,
            # 115.8,
            118.1,
            # 119.2,
            120.0,
            # 121.,
            122,
        ],
    )

    # my_ploter.create(save_figure=True, bag_name=bag_name)
    my_ploter.create(save_figure=True, bag_name=bag_name, do_subplots=False,
                     time_text=True,
                     # draw_laserscan=True,
                     )

    # my_ploter.ax.set_xlim([-7.4, 2.4])
    # my_ploter.ax.set_ylim([-4, 4])


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    bag_dir = "../data_qolo/indoor_with_david_2022_01/"
    bag_name = "2022-01-26-17-50-23.bag"

    import_again = False
    if import_again or not "my_bag" in locals():
        my_bag = rosbag.Bag(bag_dir + bag_name)

    main(my_bag, bag_name=bag_name)
