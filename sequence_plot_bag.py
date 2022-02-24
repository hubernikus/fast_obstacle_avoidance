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

from fast_obstacle_avoidance.laserscan_utils import import_first_scans, reset_laserscan


class LaserscanWalkerSinglePlot:
    # def setup(
    def __init__(
        self, static_laserscan, initial_dynamics, robot, plot_times=None,
        x_lim=[-3, 4], y_lim=[-3, 3], it_max=1000, dt_simulation=0.1,
        figsize=(12, 8),
        draw_fancy_robot=False,
    ):
        self.robot = robot
        self.initial_dynamics = initial_dynamics
        self.fast_avoider = FastLidarAvoider(robot=self.robot)
        self.static_laserscan = static_laserscan

        self.it_max = it_max
        self.dt_simulation = dt_simulation
        self.draw_fancy_robot = draw_fancy_robot

        if plot_times is None:
            # Put very high to run for long...
            self.plot_times = [1e30]
        else:
            self.plot_times = plot_times

        self.fig, self.ax = plt.subplots()

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
        

    def run(self):
        """Update robot and position."""
        self.position_list[:, 0] = self.robot.pose.position
        
        it_count = 0
        while it_count < self.it_max:
            it_count += 1
            
            self.initial_velocity = self.initial_dynamics.evaluate(self.robot.pose.position)

            temp_scan = reset_laserscan(self.static_laserscan, self.robot.pose.position)

            t_start = timer()
            self.fast_avoider.update_reference_direction(temp_scan, in_robot_frame=False)
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
        self.ax.plot(
            self.position_list[0, :it_end],
            self.position_list[1, :it_end], '--',
            color='#4a6bc8', linewidth=3, zorder=-1)

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
                    s=1.0,
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

        quiver_scale = 10
        self.ax.quiver(
            self.sample_points[0, :],
            self.sample_points[1, :],
            self.inital_velocity_list[0, :],
            self.inital_velocity_list[1, :],
            scale=quiver_scale,
            color="#008080",
            label="Initial velocity",
        )

        self.ax.quiver(
            self.sample_points[0, :],
            self.sample_points[1, :],
            self.modulated_velocity_list[0, :],
            self.modulated_velocity_list[1, :],
            scale=quiver_scale,
            # color='#213970',
            color="#000080",
            label="Modulated velocity",
        )

        self.ax.legend()
        # self.ax.grid()
                
        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)
        self.ax.set_aspect("equal")

        
    def plot_robot(self, ii):
        global_ctrl_point = self.robot.pose.transform_position_from_local_to_reference(
            self.robot.control_points[:, 0]
        )

        self.sample_points.append(global_ctrl_point)
        self.inital_velocity_list.append(self.initial_velocity)
        self.modulated_velocity_list.append(self.modulated_velocity)

        if self.draw_fancy_robot:
            self.robot.plot_robot(self.ax)
        else:
            self.robot.plot2D(self.ax, length_x=0.9)
        


def multi_plot_static_data_narrow_doorway(save_plot=False):
    bag_name = '2021-12-13-18-33-06.bag'
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
        static_laserscan=qolo.get_allscan(), initial_dynamics=dynamical_system,
        plot_times=plot_times,
        it_max=160,
        robot=qolo,
        x_lim=x_lim, y_lim=y_lim,
        figsize=(4, 3),
        )
    
    main_plotter.run()

    if save_plot:
        figure_name = "multi_plot_door_passing"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def multiplo_lab_enviornment_nice_qolo(save_plot=False):
    bag_dir = "../data_qolo/indoor_with_david_2022_01/"
    bag_name = "2022-01-26-17-50-23.bag"
    
    # eval_time = 1643215823507522977
    eval_time = 1643215823.507523 + 46.6

    x_lim = [-4, 6]
    y_lim = [-2.5, 5]
    
    # start_position = np.array([1.0, -0.4])
    # attractor_position = np.array([-1.7, 1.6])
    start_position = np.array([-1.1, 0.3])
    attractor_position = np.array([4.83, 2.34])

    plot_times = [15, 40, 70,]

    qolo = QoloRobot(
        pose=ObjectPose(position=start_position, orientation=0 * np.pi / 180)
    )

    import_first_scans(qolo, bag_name, bag_dir=bag_dir,
                       start_time=eval_time, save_intensity=True)

    fast_avoider = FastLidarAvoider(robot=qolo, evaluate_normal=True)

    dynamical_system = LinearSystem(
        attractor_position=attractor_position,
        maximum_velocity=0.8,
    )
    
    main_plotter = LaserscanWalkerSinglePlot(
        static_laserscan=qolo.get_allscan(), initial_dynamics=dynamical_system,
        plot_times=plot_times,
        it_max=160,
        robot=qolo,
        x_lim=x_lim, y_lim=y_lim,
        figsize=(4, 3),
        draw_fancy_robot=True
        )
    
    main_plotter.run()

    if save_plot:
        figure_name = "multi_fancyplot_room_passing"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")
    
        

if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # multi_plot_static_data_narrow_doorway(save_plot=True)
    multiplo_lab_enviornment_nice_qolo(save_plot=True)
    
    pass
