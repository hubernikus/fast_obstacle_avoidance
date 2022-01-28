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

            if self.start_time is None:
                self.start_time = t.to_sec()

            time_rel = t.to_sec() - self.start_time

            if time_rel > self.simulation_time:
                yield 0

            if topic == "/front_lidar/scan" or topic == "/rear_lidar/scan":
                self.robot.set_laserscan(msg, topic_name=topic, save_intensity=True)

            if topic == "/rwth_tracker/tracked_persons":
                breakpoint
                # msg_persons = msg

            if topic == "/qolo/pose2D":
                msg_qolo = msg
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

        # All done - no iteration possible anymore
        yield 1

    def setup(
        self,
        robot,
        my_bag,
        x_lim=[-7.5, 0],
        y_lim=[-4.0, 4.0],
        position_storing_length=50,
    ):
        self.robot = robot

        self.it_pos = 0

        self.static_laserscan = None

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.obstacle_color = np.array([177, 124, 124]) / 255.0

        self.x_lim = x_lim
        self.y_lim = y_lim

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
        )

        self.ax.text(
            self.x_lim[1] - 0.1 * (self.x_lim[1] - self.x_lim[0]),
            self.y_lim[1] - 0.05 * (self.y_lim[1] - self.y_lim[0]),
            f"{str(round(self.simulation_time, 2))} s",
            fontsize=14,
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

        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
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


def evaluate_run_in_room(
    my_bag,
    x_lim,
    y_lim,
    t_max=10,
    dt_simulation=0.1,
    animation_name=None,
    save_animation=False,
    bag_name=None,
    bag_dir=None,
):
    bag_path = bag_dir + bag_name
    # result = subprocess.run([f'rosbag info {bag_path} | grep duration'], stdout=subprocess.PIPE)
    # result = subprocess.run([f'rosbag info {bag_path}'], stdout=subprocess.PIPE)
    result = subprocess.run([f"rosbag info {bag_path}"], stdout=subprocess.PIPE)
    # path = subprocess.run([f'pwd'], stdout=subprocess.PIPE)

    breakpoint()
    qolo = qolo = QoloRobot(
        pose=ObjectPose(position=np.array([0, 0]), orientation=0 * np.pi / 180)
    )

    if animation_name is not None:
        now = datetime.datetime.now()
        animation_name = f"{animation_name}_{now:%Y-%m-%d_%H-%M-%S}"

    replayer = ReplayQoloCording(
        it_max=int(t_max / dt_simulation),
        dt_simulation=0.1,
        animation_name=animation_name,
    )

    replayer.setup(
        robot=qolo,
        my_bag=my_bag,
        x_lim=x_lim,
        y_lim=y_lim,
    )

    replayer.run(save_animation=save_animation)
    # replayer.run(save_animation=True)


first_simulation_options = {
    "bag_dir": "../data_qolo/indoor_working/",
    "bag_name": "2022-01-24-18-33-30.bag",
    "x_lim": [-9.5, 1.5],
    "y_lim": [-4.0, 4.0],
    "t_max": 58,
    "dt_simulation": 0.1,
    "animation_name": "animation_first_indoor",
}

second_simulation_options = {
    "bag_dir": "../data_qolo/indoor_working/",
    "bag_name": "2022-01-24-18-34-48.bag",
    "x_lim": [-9.0, 2.0],
    "y_lim": [-3.0, 5.0],
    "t_max": 15,
    "dt_simulation": 0.1,
    "animation_name": "animation_second_indoor",
}

third_simulation_options = {
    "bag_dir": "../data_qolo/indoor_working/",
    "bag_name": "2022-01-26-18-21-31.bag",
    "x_lim": [-9.0, 3.0],
    "y_lim": [-3.0, 5.0],
    "t_max": 100,
    "dt_simulation": 0.1,
    "animation_name": "animation_upperbody_first",
}


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    save_animation = False
    # my_simu = first_simulation_options
    # my_simu = second_simulation_options
    my_simu = third_simulation_options

    reimport_bag = False
    if (
        reimport_bag
        or not "my_bag" in locals()
        or not my_bag_name == my_simu["bag_name"]
    ):

        my_bag = rosbag.Bag(my_simu["bag_dir"] + my_simu["bag_name"])
        my_bag_name = my_simu["bag_name"]

    evaluate_run_in_room(my_bag, save_animation=save_animation, **my_simu)
