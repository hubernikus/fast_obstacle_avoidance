""" Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2021-12-14
# Email: lukas.huber@epfl.ch

import sys
import os
from timeit import default_timer as timer

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


class LaserScanAnimator(Animator):
    def setup(
        self, static_laserscan, initial_dynamics, robot, x_lim=[-3, 4], y_lim=[-3, 3]
    ):
        self.robot = robot
        self.initial_dynamics = initial_dynamics
        self.fast_avoider = FastLidarAvoider(robot=self.robot)
        self.static_laserscan = static_laserscan

        self.fig, self.ax = plt.subplots(figsize=(12, 8))

        self.obstacle_color = np.array([177, 124, 124]) / 255.0

        self.x_lim = x_lim
        self.y_lim = y_lim

        dimension = 2
        self.position_list = np.zeros((dimension, self.it_max + 1))

    def update_step(self, ii):
        """Update robot and position."""
        initial_velocity = self.initial_dynamics.evaluate(self.robot.pose.position)
        
        start = timer()
        temp_scan = reset_laserscan(self.static_laserscan, self.robot.pose.position)
        
        self.fast_avoider.update_reference_direction(temp_scan, in_robot_frame=False)
        modulated_velocity = self.fast_avoider.avoid(initial_velocity)

        print('pose', self.robot.pose.position)
        print('init vel', initial_velocity)
        print('mod vel', modulated_velocity)

        end = timer()
        
        print(
            "Time for modulation {}ms at it={}".format(
                np.round((end - start) * 1000, 3), ii
            )
        )

        # Update qolo
        self.robot.pose.position = (
            self.robot.pose.position + self.dt_simulation * modulated_velocity
        )
        
        if LA.norm(modulated_velocity):
            self.robot.pose.orientation = np.arctan2(
                modulated_velocity[1], modulated_velocity[0]
            )

        # self.robot.pose.orientation += self.dt_simulation*modulated_velocity
        self.position_list[:, ii] = self.robot.pose.position

        self.ax.clear()
        # Plot
        # self.ax.plot(self.robot.pose.position[0],
        # self.robot.pose.position[1], 'o', color='k')

        self.ax.plot(
            self.static_laserscan[0, :],
            self.static_laserscan[1, :],
            ".",
            color=self.obstacle_color,
            zorder=-1,
        )

        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.set_aspect("equal")

        global_ctrl_point = self.robot.pose.transform_position_from_local_to_reference(
            self.robot.control_points[:, 0]
        )

        arrow_scale = 0.4
        self.ax.arrow(
            global_ctrl_point[0],
            global_ctrl_point[1],
            arrow_scale * initial_velocity[0],
            arrow_scale * initial_velocity[1],
            width=0.03,
            head_width=0.2,
            color="g",
            label="Initial",
        )

        self.ax.arrow(
            global_ctrl_point[0],
            global_ctrl_point[1],
            arrow_scale * modulated_velocity[0],
            arrow_scale * modulated_velocity[1],
            width=0.03,
            head_width=0.2,
            color="b",
            label="Modulated",
        )

        self.robot.plot2D(self.ax)
        self.ax.plot(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            "*",
            color="black",
        )
        # print(tt)
        self.ax.grid()

    def has_converged(self, ii):
        conv_margin = 1e-4
        if (
            LA.norm(self.position_list[:, ii] - self.position_list[:, ii - 1])
            < conv_margin
        ):
            return True
        else:
            return False


def get_topics(rosbag_name):
    import bagpy

    # get the list of topics
    print(bagpy.bagreader(rosbag_name).topic_table)


def static_plot(allscan, qolo, dynamical_system, fast_avoider):
    initial_velocity = dynamical_system.evaluate(qolo.pose.position)
    modulated_velocity = fast_avoider.avoid(initial_velocity, allscan)

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(0, 0, "o", color="k")
    # ax.plot(frontscan[0, :], frontscan[1, :], '.', color='r')
    # ax.plot(rearscan[0, :], rearscan[1, :], '.', color='b')
    ax.plot(allscan[0, :], allscan[1, :], ".", color="k")

    ax.set_xlim([-10, 5])
    ax.set_ylim([-5, 5])
    ax.set_aspect("equal")

    ax.arrow(
        qolo.pose.position[0],
        qolo.pose.position[1],
        initial_velocity[0],
        initial_velocity[1],
        width=0.05,
        head_width=0.3,
        color="g",
        label="Initial",
    )

    ax.arrow(
        qolo.pose.position[0],
        qolo.pose.position[1],
        modulated_velocity[0],
        modulated_velocity[1],
        width=0.05,
        head_width=0.3,
        color="b",
        label="Modulated",
    )

    ax.legend()

    qolo.plot2D(ax)
    # print(tt)
    ax.grid()
    # for topic, msg, t in my_bag.read_messages(topics=[
    # '/front_lidar/scan',
    # '/rear_lidar/scan'
    # ]):
    # breakpoint()


def main_animator(
    # bag_name='2021-12-13-18-33-06.bag',
    # bag_name="2021-12-21-14-21-00.bag",
    # bag_name="2021-12-23-18-23-16.bag",
    bag_name="2021-12-23-18-23-16.bag",
    eval_time=1640280207.915730,
    start_position=None,
    attractor_position=None,
):
    # sample_freq = 20
    # allscan = allscan[:,  np.logical_not(np.mod(np.arange(allscan.shape[1]), sample_freq))]

    if start_position is None:
        start_position = np.array([0, 0.0])

    if attractor_position is None:
        attractor_position = np.array([0, 1])
    qolo = QoloRobot(
        pose=ObjectPose(position=start_position, orientation=0 * np.pi / 180)
    )

    import_first_scans(qolo, bag_name, start_time=eval_time)

    fast_avoider = FastLidarAvoider(robot=qolo, evaluate_normal=True)

    dynamical_system = LinearSystem(
        # attractor_position=np.array([-2, 2]),
        attractor_position=attractor_position,
        maximum_velocity=0.8,
    )

    main_animator = LaserScanAnimator(
        it_max=160, dt_simulation=0.04,
        # animation_name="indoor_scattered"
    )
    main_animator.setup(
        static_laserscan=qolo.get_allscan(),
        initial_dynamics=dynamical_system,
        robot=qolo,
    )

    main_animator.run(save_animation=True)
    # main_animator.run(save_animation=True)
    # main_animator.update_step(ii=0)


def animator_office_room():
    main_animator(bag_name="2021-12-23-18-23-16.bag",
                  start_position=np.array([-0.5, 1.5]),
                  attractor_position=np.array([3, -1])
                  )


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # main_animator()
    animator_office_room()

    pass
