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

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from fast_obstacle_avoidance.control_robot import QoloRobot
from fast_obstacle_avoidance.utils import laserscan_to_numpy
from fast_obstacle_avoidance.obstacle_avoider import (
    FastObstacleAvoider,SampledAvoider, MixedEnvironmentAvoider
    )

from fast_obstacle_avoidance.laserscan_utils import import_first_scans, reset_laserscan
from fast_obstacle_avoidance.laserscan_utils import import_first_scan_and_crowd


def is_interesecting(obstacle_environment, position):
    for obs in obstacle_environment:
        gamma = obs.get_gamma(position, in_global_frame=True)
        if gamma < 1:
            return True
    return False


def run_vectorfield_mixed(qolo):
    """Draw the vectorfield mixed"""
    qolo._got_new_obstacles = True

    my_avoider = MixedEnvironmentAvoider(qolo)
    qolo.obstacle_environment.update_reference_points()
    
    my_avoider.update_laserscan(qolo.get_allscan())
    my_avoider.update_reference_direction()
    

    # x_lim = [-3, 4]
    # y_lim = [-3, 3]
    limit_scan = my_avoider.get_scan_without_ocluded_points()

    x_lim = [-7.5, 7.5]
    y_lim = [-7.5, 7.5]

    dynamical_system = LinearSystem(
        # attractor_position=np.array([1.5, 0.5]),
        # attractor_position=np.array([-1.5, 1.5]),
        attractor_position=np.array([6.0, 4.0]),
        maximum_velocity=0.8,
    )

    nx = ny = 120

    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities_init = np.zeros(positions.shape)
    velocities_mod = np.zeros(positions.shape)
    reference_dirs = np.zeros(positions.shape)

    for it in range(positions.shape[1]):
        qolo.pose.position = positions[:, it]
        qolo._got_new_obstacles = True
        temp_scan = reset_laserscan(qolo.get_allscan(), positions[:, it])

        _, _, relative_distances = qolo.get_relative_positions_and_dists(
            temp_scan, in_robot_frame=False
        )
        
        if any(relative_distances < 0):
            continue

        if is_interesecting(qolo.obstacle_environment, positions[:, it]):
            continue

        my_avoider.update_laserscan(temp_scan)
        my_avoider.update_reference_direction(in_robot_frame=False)

        velocities_init[:, it] = dynamical_system.evaluate(positions[:, it])
        velocities_mod[:, it] = my_avoider.avoid(velocities_init[:, it])
        reference_dirs[:, it] = my_avoider.reference_direction


    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(
        my_avoider.laserscan[0, :],
        my_avoider.laserscan[1, :],
        ".",
        # color=np.array([177, 124, 124]) / 255.0,
        color="black",
        alpha=0.2,
        zorder=-1,
    )

    ax.plot(
        limit_scan[0, :],
        limit_scan[1, :],
        ".",
        # color=np.array([177, 124, 124]) / 255.0,
        # color='b',
        color="black",
        zorder=-1,
    )

    plot_obstacles(
        ax,
        my_avoider.obstacle_avoider.obstacle_environment,
        showLabel=False,
        draw_reference=True,
        velocity_arrow_factor=1.0,
        # noTicks=True,
        x_lim=x_lim,
        y_lim=y_lim,
    )

    plot_quiver = False
    if plot_quiver:
        arrow_width = 0.003
        scale_vel = 2  #
        
        if scale_vel is not None:
            ax.quiver(
            positions[0, :],
            positions[1, :],
            velocities_init[0, :],
            velocities_init[1, :],
            angles='xy', scale_units='xy',
            scale=scale_vel,
            width=arrow_width,
            color="black",
            alpha=0.3,
            )

            # ax.quiver(
                # positions[0, :],
                # positions[1, :],
                # velocities_mod[0, :],
                # velocities_mod[1, :],
                # angles="xy",
                # scale_units="xy",
                # scale=scale_vel,
                # width=arrow_width,
                # color="blue",
            # )
    else:
        ax.streamplot(
            positions[0, :].reshape(nx, ny),
            positions[1, :].reshape(nx, ny),
            velocities_mod[0, :].reshape(nx, ny),
            velocities_mod[1, :].reshape(nx, ny),
            # angles="xy",
            # scale_units="xy",
            # scale=scale_vel,
            # width=arrow_width,
            color="blue",
        )

    ax.plot(
        dynamical_system.attractor_position[0],
        dynamical_system.attractor_position[1],
        "k*",
        linewidth=13.0,
        markersize=12,
        zorder=5,
    )

    ax.grid()

    plt.ion()
    plt.show()

    # breakpoint()
    figure_name = "mixed_environment_streamplot"
 
    if figure_name:
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight", dpi=300)


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # run_vectorfield_mixed()
    
    do_the_import = False
    # do_the_import = False
    if not 'qolo' in vars() or not 'qolo' in globals() or do_the_import:
        qolo = QoloRobot(
            pose=ObjectPose(position=[0.7, -0.7], orientation=30 * np.pi / 180)
        )

        import_first_scan_and_crowd(
            robot=qolo,
            bag_name="2021-12-03-18-21-29.bag",
            bag_dir="/home/lukas/Code/data_qolo/outdoor_recording/",
            start_time=None,
        )

    run_vectorfield_mixed(qolo=qolo)
