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


def main_vectorfield(
    figure_name="vector_field_around_laserscan",
    bag_name="2021-12-13-18-33-06.bag",
    # bag_name="2021-12-21-14-21-00.bag",
    # bag_name="2021-12-23-18-23-16.bag",
    eval_time=1640280207.915730,
):
    nx = ny = 25

    qolo = QoloRobot(
        pose=ObjectPose(position=[0.7, -0.7], orientation=30 * np.pi / 180)
    )

    import_first_scans(qolo, bag_name, start_time=None)

    allscan = qolo.get_allscan()

    # fast_avoider = FastLidarAvoider(robot=qolo, evaluate_normal=True)
    fast_avoider = FastLidarAvoider(
        robot=qolo,
        evaluate_normal=False,
        weight_max_norm=1e4,
    )

    dynamical_system = LinearSystem(
        # attractor_position=np.array([1.5, 0.5]),
        attractor_position=np.array([-1.5, 1.5]),
        # attractor_position=np.array([2.0, 0.0]),
        maximum_velocity=0.8,
    )

    # pos = np.array([4., 3.])
    # temp_scan = reset_laserscan(allscan, pos)
    # fast_avoider.update_reference_direction(temp_scan)
    # vel_init = dynamical_system.evaluate(pos)
    # vel_mod = fast_avoider.avoid(vel_init)

    # print("vel_init", vel_init)
    # print("vel_mod", vel_mod)
    # print("fast_avoider", fast_avoider.reference_direction)

    # if True:
    # return

    x_lim = [-3, 4]
    y_lim = [-3, 3]

    # x_lim = [0.4, 1.2]
    # y_lim = [-0.3, 0.5]

    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities_init = np.zeros(positions.shape)
    velocities_mod = np.zeros(positions.shape)
    reference_dirs = np.zeros(positions.shape)

    it_count = 0
    t_sum = 0

    for it in range(positions.shape[1]):
        qolo.pose.position = positions[:, it]
        temp_scan = reset_laserscan(allscan, positions[:, it])

        _, _, relative_distances = qolo.get_relative_positions_and_dists(
            temp_scan, in_robot_frame=False
        )

        if any(relative_distances < 0):
            continue

        velocities_init[:, it] = dynamical_system.evaluate(positions[:, it])

        temp_scan = np.repeat(temp_scan, 30, axis=1)
        print(f"[WARNING]: Repeating the scan to shape={temp_scan.shape}.")

        t_start = timer()
        # fast_avoider.update_reference_direction(temp_scan, in_robot_frame=False)
        fast_avoider.update_laserscan(temp_scan, in_robot_frame=False)
        # fast_avoider.update_reference_direction()
        velocities_mod[:, it] = fast_avoider.avoid(velocities_init[:, it])
        t_end = timer()
        print(f"Time modulation {np.round((t_end-t_start) * 1000, 3)}ms")

        t_sum += t_end - t_start
        it_count += 1

        reference_dirs[:, it] = fast_avoider.reference_direction

    print(f"Average time of {t_sum / it_count * 1000}")
    fig, axs = plt.subplots(1, 2, figsize=(10, 6))

    # ax.quiver(positions[0, :], positions[1, :],
    # velocities[0, :], velocities[1, :],
    # color='black', alpha=0.3)
    for ax in axs:
        ax.plot(
            allscan[0, :],
            allscan[1, :],
            ".",
            color=np.array([177, 124, 124]) / 255.0,
            zorder=-1,
        )

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_aspect("equal")

        ax.grid(True)

    arrow_width = 0.004

    axs[0].quiver(
        positions[0, :],
        positions[1, :],
        reference_dirs[0, :],
        reference_dirs[1, :],
        # angles='xy', scale_units='xy', scale=scale_vel,
        scale=30,
        color="black",
        width=arrow_width,
        alpha=0.8,
    )

    # axs[0].title("Reference Vectors")
    scale_vel = 3  #
    if scale_vel is not None:
        axs[1].quiver(
            positions[0, :],
            positions[1, :],
            velocities_init[0, :],
            velocities_init[1, :],
            angles="xy",
            scale_units="xy",
            scale=scale_vel,
            width=arrow_width,
            color="black",
            alpha=0.3,
        )

        axs[1].quiver(
            positions[0, :],
            positions[1, :],
            velocities_mod[0, :],
            velocities_mod[1, :],
            angles="xy",
            scale_units="xy",
            scale=scale_vel,
            width=arrow_width,
            color="blue",
        )
    else:
        # axs[1].quiver(
        # positions[0, :],
        # positions[1, :],
        # velocities_init[0, :],
        # velocities_init[1, :],
        # scale=30,
        # color="green",
        # )

        axs[1].quiver(
            positions[0, :],
            positions[1, :],
            velocities_mod[0, :],
            velocities_mod[1, :],
            scale=30,
            color="blue",
        )

    axs[1].plot(
        dynamical_system.attractor_position[0],
        dynamical_system.attractor_position[1],
        "k*",
        linewidth=13.0,
        markersize=12,
        zorder=5,
    )

    nx = ny = 10
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    deviation = np.zeros((positions.shape[1]))

    for it in range(positions.shape[1]):
        qolo.pose.position = positions[:, it]
        temp_scan = reset_laserscan(allscan, positions[:, it])

        _, _, relative_distances = qolo.get_relative_positions_and_dists(temp_scan)

        if any(relative_distances < 0):
            continue

        fast_avoider.update_reference_direction(temp_scan, in_robot_frame=False)

        ref_dirs = fast_avoider.reference_direction

        if fast_avoider.normal_direction is not None:
            norm_dirs = fast_avoider.normal_direction
            deviation[it] = np.arcsin(np.cross(ref_dirs, norm_dirs))

            # breakpoint()
    if fast_avoider.normal_direction:
        pcm = axs[0].contourf(
            x_vals,
            y_vals,
            deviation.reshape(nx, ny),
            # cmap='PiYG',
            cmap="bwr",
            vmin=-np.pi / 2,
            vmax=np.pi / 2,
            # vmin=-0.7, vmax=0.7,
            zorder=-3,
            alpha=0.9,
            levels=101,
        )

        cbar = fig.colorbar(
            pcm,
            ax=axs[0],
            fraction=0.035,
            ticks=[-0.5, 0, 0.5],
            extend="neither",
        )


if (__name__) == "__main__":
    # plt.close("all")
    plt.ion()

    # vectorfield_inside()
    main_vectorfield()
