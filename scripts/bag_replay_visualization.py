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

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from fast_obstacle_avoidance.control_robot import QoloRobot
from fast_obstacle_avoidance.utils import laserscan_to_numpy
from fast_obstacle_avoidance.obstacle_avoider import FastLidarAvoider
from fast_obstacle_avoidance.laserscan_utils import import_first_scans


class ReplayQoloCording(Animator):
    def rosbag_generator(self, my_bag, bag_dir=None, bag_name=None, dx_max=1):
        """Generator member function which allows state-storing and updating.
        Yields/Return 0 for success; all data is stored in `self`."""
        if my_bag is None:
            rosbag_name = bag_dir + bag_name
            # my_bag = bagpy.bagreader(rosbag)
            # my_bag = bagpy.bagreader(rosbag_name)
            my_bag = rosbag.Bag(rosbag_name)

        self.start_time = None
        self.pos_start = None

        for topic, msg, t in my_bag.read_messages(topics=self.topic_names):

            if self.start_time is None:
                self.start_time = t.to_sec()

            time_rel = t.to_sec() - self.start_time

            if time_rel > self.simulation_time:
                yield 0

            if topic == "/front_lidar/scan" or topic == "/rear_lidar/scan":
                self.robot.set_laserscan(msg, topic_name=topic, save_intensity=True)

            if topic == "/qolo/pose2D":
                self.robot.pose.position = np.array([msg.x, msg.y])
                self.robot.pose.orientation = msg.theta

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

            if topic == "/rwth_tracker/tracked_persons":
                breakpoint()
                self.robot.set_crowdtracker(msg)

        # All done - no iteration possible anymore
        yield 1

    def setup(
        self,
        robot,
        my_bag,
        x_lim=None,
        y_lim=None,
        plot_width_x=11,
        plot_width_y=8,
        position_storing_length=50,
        topic_names=None,
    ):
        self.robot = robot

        self.it_pos = 0

        self.static_laserscan = None

        if topic_names is None:
            self.topic_names = [
                "/front_lidar/scan",
                "/rear_lidar/scan",
                "/rwth_tracker/tracked_persons",
                "/qolo/user_commands",  # -> input
                "/qolo/remote_commands",  # -> output
                # '/qolo/twist', # (? and this?)
                "/qolo/pose2D",
                "/rwth_tracker/tracked_persons",
            ]
        else:
            self.topic_names = topic_names

        self.fig, self.ax = plt.subplots(figsize=(16, 10))
        self.obstacle_color = np.array([177, 124, 124]) / 255.0

        self.x_lim = x_lim
        self.y_lim = y_lim

        # Only used if no x_lim / y_lim given
        self.width_x = plot_width_x
        self.width_y = plot_width_y

        self.dimension = 2
        # self.position_list = np.zeros((self.dimension, position_storing_length))

        self.initial_velocity = np.zeros(self.dimension)
        self.modulated_velocity = np.zeros(self.dimension)

        # Initialize generator and run once
        self.simulation_time = (1) * self.dt_simulation
        self.my_generator = self.rosbag_generator(my_bag)
        self.ros_state = next(self.my_generator)

        self.position_list = np.tile(
            self.robot.pose.position, (position_storing_length, 1)
        ).T

    def update_step(self, ii):
        """Do one update step."""
        self.ii = ii
        self.simulation_time = (self.ii + 1) * self.dt_simulation

        if self.ros_state:
            # All iterations done
            return 1

        self.ros_state = next(self.my_generator)

        global_ctrl_point = np.zeros(self.dimension)

        # Update position list
        self.position_list = np.roll(self.position_list, (-1), axis=1)
        self.position_list[:, -1] = self.robot.pose.position

        self.ax.clear()

        # Plot
        # self.ax.plot(
        # self.position_list[0, :self.it_pos],
        # self.position_list[1, :self.it_pos],
        # self.position_list[0, :],
        # self.position_list[1, :],
        # color='k')
        # TODO: fading line

        plt.scatter(
            self.position_list[0, :],
            self.position_list[1, :],
            c=np.arange(self.position_list.shape[1]),
            marker=".",
            s=30,
            alpha=0.6,
        )

        laserscan = self.robot.get_allscan(in_robot_frame=False)

        if laserscan.shape[1]:
            intensities = self.robot.get_all_intensities()
            self.ax.scatter(
                laserscan[0, :],
                laserscan[1, :],
                # c='black',
                c=intensities,
                cmap="copper",
                # cmap='hot',
                # ".",
                # color=self.obstacle_color,
                s=10.0,
                # alpha=(intensities/255.),
                # alpha=(1-intensities/255.),
                alpha=0.8,
                zorder=-1,
            )

        if self.x_lim is None or self.y_lim is None:
            # Just define both if one is not defined
            sensor_center = np.mean(laserscan, axis=1)

            self.x_lim = [
                sensor_center[0] - self.width_x / 2,
                sensor_center[0] + self.width_x / 2,
            ]

            self.y_lim = [
                sensor_center[1] - self.width_y / 2,
                sensor_center[1] + self.width_y / 2,
            ]

        # Check offset and potentially adapt
        fraction_to_adapt_plot_limits = 1.0 / 3
        if abs(
            np.mean(self.x_lim) - self.robot.pose.position[0]
        ) > fraction_to_adapt_plot_limits * (self.x_lim[1] - self.x_lim[0]):
            sensor_center = np.mean(laserscan, axis=1)

            self.x_lim = [
                sensor_center[0] - self.width_x / 2,
                sensor_center[0] + self.width_x / 2,
            ]

        if abs(
            np.mean(self.y_lim) - self.robot.pose.position[1]
        ) > fraction_to_adapt_plot_limits * (self.y_lim[1] - self.y_lim[0]):
            sensor_center = np.mean(laserscan, axis=1)

            self.y_lim = [
                sensor_center[1] - self.width_y / 2,
                sensor_center[1] + self.width_y / 2,
            ]

        # Plot obstacles (or at least limit environment)
        if (
            self.robot.obstacle_environment is None
            or (self.robot.obstacle_environment) == 0
        ):
            self.ax.set_xlim(self.x_lim)
            self.ax.set_ylim(self.y_lim)
            self.ax.set_aspect("equal")

        else:
            plot_obstacles(
                self.ax,
                self.robot.obstacle_environment,
                showLabel=False,
                draw_reference=True,
                velocity_arrow_factor=1.0,
                # noTicks=True,
                x_lim=self.x_lim,
                y_lim=self.y_lim,
            )

        self.ax.text(
            self.x_lim[1] - 0.1 * (self.x_lim[1] - self.x_lim[0]),
            self.y_lim[1] - 0.05 * (self.y_lim[1] - self.y_lim[0]),
            f"{str(round(self.simulation_time, 2))} s",
            fontsize=14,
        )

        self.ax.set_aspect("equal")
        self.ax.grid(zorder=-3.0)

        # self.ax.grid
        self.robot.plot_robot(self.ax)

        drawn_arrow = False
        arrow_scale = 0.2
        margin_velocity_plot = 1e-3
        if LA.norm(self.initial_velocity) > margin_velocity_plot:
            self.ax.arrow(
                self.robot.pose.position[0] + global_ctrl_point[0],
                self.robot.pose.position[1] + global_ctrl_point[1],
                arrow_scale * self.initial_velocity[0],
                arrow_scale * self.initial_velocity[1],
                width=0.03,
                head_width=0.2,
                color="g",
                label="Initial Command",
            )
            drawn_arrow = True

        if LA.norm(self.modulated_velocity) > margin_velocity_plot:
            self.ax.arrow(
                self.robot.pose.position[0] + global_ctrl_point[0],
                self.robot.pose.position[1] + global_ctrl_point[1],
                arrow_scale * self.modulated_velocity[0],
                arrow_scale * self.modulated_velocity[1],
                width=0.03,
                head_width=0.2,
                color="b",
                label="Modulated Velocity",
            )
            drawn_arrow = True

        if drawn_arrow:
            self.ax.legend(loc="upper left")

    def has_converged(self, ii):
        """ROS-state indicates"""
        return self.ros_state

    def callback_iterator(self):
        pass


def evaluate_bag(
    my_bag,
    x_lim=None,
    y_lim=None,
    t_max=10,
    dt_simulation=0.1,
    animation_name=None,
    save_animation=False,
    bag_name=None,
    bag_dir=None,
    plot_width_x=None,
    plot_width_y=None,
):
    bag_path = bag_dir + bag_name

    # Get duration
    stream = os.popen(f"rosbag info {bag_path} | grep duration")
    output = stream.read()

    duration = 0
    duration_str = ""
    for m in output:
        if len(duration_str) > 0 or m.isdigit():
            if m == "s":
                break

            elif m == ":":
                # Transform minutes to seconds [rest the minute string]
                duration = 60 * float(duration_str)
                duration_str = ""
                continue

            duration_str = duration_str + m
    duration = duration + float(duration_str)

    it_max = int(duration / dt_simulation)

    qolo = QoloRobot(
        pose=ObjectPose(position=np.array([0, 0]), orientation=0 * np.pi / 180)
    )

    if animation_name is None:
        animation_name = f"animation_bag_{bag_name[:-4]}"

    replayer = ReplayQoloCording(
        it_max=it_max,
        dt_simulation=dt_simulation,
        animation_name=animation_name,
    )

    replayer.setup(
        robot=qolo,
        my_bag=my_bag,
        x_lim=x_lim,
        y_lim=y_lim,
        plot_width_x=plot_width_x,
        plot_width_y=plot_width_y,
    )

    replayer.run(save_animation=save_animation)
    # replayer.run(save_animation=True)


def evaluate_multibags_outdoor_all(
    bag_dir="../data_qolo/marketplace_lausanne_2022/",
    save_animation=True,
    dt_evaluation=0.1,
    # plot_width_x=11, plot_width_y=8,
    # plot_width_x=18, plot_width_y=12,
    plot_width_x=16,
    plot_width_y=9,
    #
):
    """This script is for batch-processing of the ros-bag recording with the qolo."""
    # Use aspect ratio of 16:9 [youtube standard ratio] for x_lim:y_lim
    # bag_list = glob.glob(bag_dir + "/*.bag")
    bag_list = os.listdir(bag_dir)

    for bag_name in bag_list:
        my_bag = rosbag.Bag(bag_dir + bag_name)

        evaluate_bag(
            my_bag,
            bag_name=bag_name,
            bag_dir=bag_dir,
            save_animation=save_animation,
            dt_simulation=dt_evaluation,
            plot_width_x=plot_width_x,
            plot_width_y=plot_width_y,
        )


def evaluate_multibags_outdoor_second_take(
    bag_dir="../data_qolo/marketplace_lausanne_2022/",
    save_animation=True,
    dt_evaluation=0.1,
    # plot_width_x=11, plot_width_y=8,
    # plot_width_x=18, plot_width_y=12,
    plot_width_x=16,
    plot_width_y=9,
    #
):
    """This script is for batch-processing of the ros-bag recording with the qolo."""
    # Use aspect ratio of 16:9 [youtube standard ration] for x_lim / y_lim
    # bag_list = glob.glob(bag_dir + "/*.bag")
    bag_list = os.listdir(bag_dir)

    day_int = 3
    month_int = 2

    for bag_name in bag_list:
        if int(bag_name[5:7]) != month_int:
            continue

        if int(int(bag_name[8:10])) != day_int:
            continue

        print(f"Doing {bag_name}")
        my_bag = rosbag.Bag(bag_dir + bag_name)

        evaluate_bag(
            my_bag,
            bag_name=bag_name,
            bag_dir=bag_dir,
            save_animation=save_animation,
            dt_simulation=dt_evaluation,
            plot_width_x=plot_width_x,
            plot_width_y=plot_width_y,
        )


def evaluate_multibags_indoor(
    bag_dir="../data_qolo/indoor_with_david_2022_01/",
    # bag_dir="../data_qolo/indoor_working/",
    save_animation=True,
    dt_evaluation=0.1,
    # plot_width_x=11, plot_width_y=8,
    # plot_width_x=18, plot_width_y=12,
    plot_width_x=16,
    plot_width_y=9,
    #
):
    """This script is for batch-processing of the ros-bag recording with the qolo."""
    # Use aspect ratio of 16:9 [youtube standard ration] for x_lim / y_lim
    # bag_list = glob.glob(bag_dir + "/*.bag")
    bag_list = os.listdir(bag_dir)

    for bag_name in bag_list:
        print(f"Trying bag {bag_name}")
        my_bag = rosbag.Bag(bag_dir + bag_name)

        evaluate_bag(
            my_bag,
            bag_name=bag_name,
            bag_dir=bag_dir,
            save_animation=save_animation,
            dt_simulation=dt_evaluation,
            plot_width_x=plot_width_x,
            plot_width_y=plot_width_y,
        )


# animation_bag_2022-01-28-13-33-32

if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # evaluate_multibags_outdoor_all()
    # evaluate_multibags_outdoor()
    # evaluate_multibags_indoor_all()
    # evaluate_multibags_outdoor_second_take(save_animation=False)

    evaluate_single_bag = True
    if evaluate_single_bag:
        reimport_bag = True

        if (
            reimport_bag
            or not "my_bag" in locals()
            # or not my_bag_name == my_simu["bag_name"]
        ):
            save_animation = True
            dt_evaluation = 0.1

            plot_width_x = 16
            plot_width_y = 9

            bag_dir = "../data_qolo/marketplace_lausanne_2022/"
            bag_name = "2022-01-28-13-33-32.bag"
            my_bag = rosbag.Bag(bag_dir + bag_name)

        evaluate_bag(
            my_bag,
            bag_name=bag_name,
            bag_dir=bag_dir,
            save_animation=save_animation,
            dt_simulation=dt_evaluation,
            plot_width_x=plot_width_x,
            plot_width_y=plot_width_y,
        )
