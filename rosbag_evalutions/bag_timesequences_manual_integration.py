""" Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2021-12-14
# Email: lukas.huber@epfl.ch

import sys
import os
import copy

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

from fast_obstacle_avoidance.laserscan_utils import import_first_scans, reset_laserscan


class LaserscanWalkerSinglePlot:
    # def setup(
    def __init__(
        self,
        static_laserscan,
        initial_dynamics,
        robot,
        fast_avoider=None,
        plot_times=None,
        x_lim=[-3, 4],
        y_lim=[-3, 3],
        it_max=1000,
        dt_simulation=0.1,
        figsize=(12, 8),
        draw_fancy_robot=False,
        label_velocity=True,
        trajectory_label=None,
        fig_ax_tuple=None,
    ):
        self.robot = robot
        self.initial_dynamics = initial_dynamics
        if fast_avoider is None:
            self.fast_avoider = FastLidarAvoider(robot=self.robot)
        else:
            self.fast_avoider = fast_avoider
        self.static_laserscan = static_laserscan

        self.it_max = it_max
        self.dt_simulation = dt_simulation
        self.draw_fancy_robot = draw_fancy_robot
        self.trajectory_label = trajectory_label
        self.label_velocity = label_velocity

        if plot_times is None:
            # Put very high to run for long...
            self.plot_times = [1e30]
        else:
            self.plot_times = plot_times

        if fig_ax_tuple is None:
            self.fig, self.ax = plt.subplots(figsize=figsize)
        else:
            self.fig, self.ax = fig_ax_tuple

        # self.obstacle_color = np.array([177, 124, 124]) / 255.0

        self.x_lim = x_lim
        self.y_lim = y_lim

        dimension = 2
        self.position_list = np.zeros((dimension, self.it_max + 1))
        self.initial_velocity = np.zeros(dimension)

        # Points for platting the 'snapshots'
        self.sample_points = []
        self.inital_velocity_list = []
        self.modulated_velocity_list = []

    def run(self, basecolor=None):
        """Update robot and position."""
        self.position_list[:, 0] = self.robot.pose.position
        self.basecolor = basecolor

        it_count = 0
        while it_count < self.it_max:
            it_count += 1

            self.initial_velocity = self.initial_dynamics.evaluate(
                self.robot.pose.position
            )

            temp_scan = reset_laserscan(self.static_laserscan, self.robot.pose.position)

            t_start = timer()
            self.fast_avoider.update_reference_direction(
                temp_scan, in_robot_frame=False
            )
            self.modulated_velocity = self.fast_avoider.avoid(self.initial_velocity)
            t_end = timer()

            # Update qolo
            self.robot.pose.position = (
                self.robot.pose.position + self.dt_simulation * self.modulated_velocity
            )

            if LA.norm(self.modulated_velocity):
                self.robot.pose.orientation = np.arctan2(
                    self.modulated_velocity[1], self.modulated_velocity[0]
                )

            self.position_list[:, it_count] = self.robot.pose.position

            if not LA.norm(self.modulated_velocity):
                print("Stopped at it={it_count}")
                break

            if len(self.plot_times) and it_count > self.plot_times[0]:
                self.plot_robot(ii=it_count)
                del self.plot_times[0]

                # if not len(self.plot_times):
                # Done for all times
                # break

        self.plot_environment(it_count)

    def plot_environment(self, it_end):
        if self.basecolor is None:
            self.basecolor = "#4a6bc8"

        self.ax.plot(
            self.position_list[0, :it_end],
            self.position_list[1, :it_end],
            "--",
            color=self.basecolor,
            linewidth=3,
            zorder=-1,
            alpha=0.9,
            label=self.trajectory_label,
        )

        intensities = self.robot.get_all_intensities()
        self.ax.scatter(
            self.static_laserscan[0, :],
            self.static_laserscan[1, :],
            # c='black',
            # c=self.robot.get_all_intensities(),
            c=intensities,
            cmap="copper",
            # cmap='hot',
            # ".",
            # color=self.obstacle_color,
            s=4.0,
            # alpha=(intensities/255.),
            # alpha=(1-intensities/255.),
            alpha=0.8,
            zorder=-1,
        )
        # self.ax.plot(
        #     self.static_laserscan[0, :],
        #     self.static_laserscan[1, :],
        #     ".",
        #     color=self.obstacle_color,
        #     zorder=-1,
        # )

        self.ax.plot(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

        # Plot velocities here
        self.sample_points = np.array(self.sample_points).T
        self.inital_velocity_list = np.array(self.inital_velocity_list).T
        self.modulated_velocity_list = np.array(self.modulated_velocity_list).T

        # breakpoint()
        quiver_scale = 10
        if self.label_velocity:
            tmp_label = "Initial velocity"
        else:
            tmp_label = None
        self.ax.quiver(
            self.sample_points[0, :],
            self.sample_points[1, :],
            self.inital_velocity_list[0, :],
            self.inital_velocity_list[1, :],
            scale=quiver_scale,
            color="#008080",
            label=tmp_label,
        )

        if self.label_velocity:
            tmp_label = "Modulated velocity"
        else:
            tmp_label = None

        self.ax.quiver(
            self.sample_points[0, :],
            self.sample_points[1, :],
            self.modulated_velocity_list[0, :],
            self.modulated_velocity_list[1, :],
            scale=quiver_scale,
            # color='#213970',
            color="#000080",
            label=tmp_label,
        )

        self.ax.legend()
        # self.ax.grid()

        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.set_aspect("equal")

    def plot_robot(self, ii):
        global_ctrl_point = self.robot.pose.transform_position_from_relative(
            self.robot.control_points[:, 0]
        )

        # breakpoint()

        self.sample_points.append(global_ctrl_point)
        self.inital_velocity_list.append(self.initial_velocity)
        self.modulated_velocity_list.append(self.modulated_velocity)

        if self.draw_fancy_robot:
            self.robot.plot_robot(self.ax, length_x=1.0)
        else:
            self.robot.plot2D(self.ax, patch_color=self.basecolor)


def multi_plot_static_data_narrow_doorway(save_plot=False):
    bag_name = "2021-12-13-18-33-06.bag"
    eval_time = 0

    start_position = np.array([1.0, -0.4])
    attractor_position = np.array([-1.7, 1.6])

    plot_times = [0, 15, 30, 45]

    # save_animation=False,
    # bag_name="2021-12-23-18-23-16.bag"
    # eval_time=1640280207.915730

    # start_position=np.array([-0.5, 1.5])
    # attractor_position=np.array([3, -1])

    qolo = QoloRobot(
        pose=ObjectPose(position=start_position, orientation=0 * np.pi / 180)
    )

    import_first_scans(qolo, bag_name, start_time=eval_time, save_intensity=True)

    fast_avoider = FastLidarAvoider(robot=qolo, evaluate_normal=True)

    dynamical_system = LinearSystem(
        attractor_position=attractor_position,
        maximum_velocity=0.8,
    )
    x_lim = [-2.5, 3]
    y_lim = [-2, 2.5]

    main_plotter = LaserscanWalkerSinglePlot(
        static_laserscan=qolo.get_allscan(),
        initial_dynamics=dynamical_system,
        plot_times=plot_times,
        it_max=160,
        robot=qolo,
        x_lim=x_lim,
        y_lim=y_lim,
        # figsize=(4.0, 3.5),
        figsize=(5.0, 4.0),
    )

    main_plotter.run()

    if save_plot:
        figure_name = "multi_plot_door_passing"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def multi_plot_static_data_narrow_doorway_comparison(save_plot=False):
    bag_dir = "/home/lukas/Recordings/fast_avoidance/door_foa/"
    bag_name = "2022-11-04-12-41-12.bag"
    eval_time = 0

    relative_eval_time = 14.6

    # start_position = np.array([-3, 0.5])
    # attractor_position = np.array([-3, -4.5])

    plot_times = [20, 60]

    # x_lim = [-8, 8]
    # y_lim = [-8, 8]

    # x_lim = [-6, 4]
    # y_lim = [-3, 3.5]

    x_lim = [-5, 3]
    y_lim = [-2, 2.5]

    start_position = np.array([-4, 1.0])
    attractor_position = np.array([1.7, 1.0])

    qolo = QoloRobot(
        pose=ObjectPose(position=start_position, orientation=0 * np.pi / 180)
    )

    dynamical_system = LinearSystem(
        attractor_position=attractor_position,
        maximum_velocity=0.8,
    )

    import_first_scans(
        qolo,
        bag_name,
        bag_dir=bag_dir,
        relative_eval_time=relative_eval_time,
        save_intensity=True,
    )

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    robot = copy.deepcopy(qolo)
    fast_avoider = FastLidarAvoider(robot=robot)
    main_plotter = LaserscanWalkerSinglePlot(
        static_laserscan=qolo.get_allscan(),
        initial_dynamics=dynamical_system,
        fast_avoider=fast_avoider,
        plot_times=plot_times,
        it_max=160,
        robot=robot,
        x_lim=x_lim,
        y_lim=y_lim,
        fig_ax_tuple=(fig, ax),
        trajectory_label="FOA",
        label_velocity=False,
    )
    main_plotter.run(basecolor="#7b1fa2")

    from fast_obstacle_avoidance.comparison.vfh_avoider import VFH_Avoider

    robot = copy.deepcopy(qolo)
    vfh_avoider = VFH_Avoider(robot=robot)
    vfh_avoider.histogram_thresholds = (2, 10)
    vfh_avoider.num_angular_sectors = 180
    plot_times = [20, 59]

    dt_simu = 0.1

    vfh_plotter = LaserscanWalkerSinglePlot(
        static_laserscan=qolo.get_allscan(),
        initial_dynamics=dynamical_system,
        fast_avoider=vfh_avoider,
        plot_times=plot_times,
        # it_max=int(plot_times[-1] / dt_simu) + 1,
        it_max=plot_times[-1] + 1,
        dt_simulation=dt_simu,
        robot=robot,
        x_lim=x_lim,
        y_lim=y_lim,
        fig_ax_tuple=(fig, ax),
        trajectory_label="VFH",
    )
    vfh_plotter.run(basecolor="#AFB42B")

    # Plot final robot
    qolo.plot_robot(ax, length_x=1.0)

    if save_plot:
        figure_name = "multi_plot_comparison"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight", dpi=300)


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # multi_plot_static_data_narrow_doorway(save_plot=True)
    multi_plot_static_data_narrow_doorway_comparison(save_plot=True)

    # multiplot_lab_enviornment_nice_qolo(save_plot=True)

    pass
