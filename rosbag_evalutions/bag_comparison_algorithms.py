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
# import rosbag

from vartools.states import ObjectPose
from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from fast_obstacle_avoidance.control_robot import QoloRobot
from fast_obstacle_avoidance.utils import laserscan_to_numpy
from fast_obstacle_avoidance.obstacle_avoider import (
    SampledAvoider,
    FastObstacleAvoider,
    MixedEnvironmentAvoider,
)

from fast_obstacle_avoidance.laserscan_utils import import_first_scans, reset_laserscan
from fast_obstacle_avoidance.laserscan_utils import import_first_scan_and_crowd


def is_interesecting(obstacle_environment, position):
    for obs in obstacle_environment:
        gamma = obs.get_gamma(position, in_global_frame=True)
        if gamma < 1:
            return True
    return False


def plot_environment(
    ax,
    my_avoider,
    positions=None,
    velocities_mod=None,
    dynamical_system=None,
    x_lim=None,
    y_lim=None,
    draw_velocity_arrow=False,
    show_quiver=False,
):

    if isinstance(my_avoider, FastObstacleAvoider) or hasattr(
        my_avoider, "obstacle_avoider"
    ):
        plot_obstacles(
            ax,
            my_avoider.robot.obstacle_environment,
            showLabel=False,
            draw_reference=True,
            velocity_arrow_factor=1.0,
            drawVelArrow=draw_velocity_arrow,
            x_lim=x_lim,
            y_lim=y_lim,
        )

    if hasattr(my_avoider, "get_scan_without_ocluded_points"):
        limit_scan = my_avoider.get_scan_without_ocluded_points()
        ax.plot(
            limit_scan[0, :],
            limit_scan[1, :],
            ".",
            # color=np.array([177, 124, 124]) / 255.0,
            # color='b',
            color="black",
            zorder=-1,
        )

    if hasattr(my_avoider, "laserscan"):
        ax.plot(
            my_avoider.laserscan[0, :],
            my_avoider.laserscan[1, :],
            ".",
            # color=np.array([177, 124, 124]) / 255.0,
            color="black",
            alpha=1.0,
            zorder=-1,
        )

    if velocities_mod is not None:
        if show_quiver:
            arrow_width = 0.003
            scale_vel = 2  #

            if scale_vel is not None:
                ax.quiver(
                    positions[0, :],
                    positions[1, :],
                    velocities_mod[0, :],
                    velocities_mod[1, :],
                    angles="xy",
                    scale_units="xy",
                    scale=scale_vel,
                    width=arrow_width,
                    color="blue",
                    # alpha=0.3,
                )

        else:
            nx = ny = int(np.sqrt(positions.shape[1]))

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

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    ax.grid()
    ax.set_aspect("equal")


def visualization_sampledata(ax, qolo, dynamical_system, x_lim, y_lim, n_sampels=10):
    """Draw the vectorfield mixed"""
    qolo._got_new_obstacles = True

    my_avoider = SampledAvoider(robot=qolo)
    # qolo.obstacle_environment.update_reference_points()
    # my_avoider.update_laserscan(qolo.get_allscan())
    # my_avoider.update_reference_direction()

    nx = ny = n_sampels

    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities_init = np.zeros(positions.shape)
    velocities_mod = np.zeros(positions.shape)
    reference_dirs = np.zeros(positions.shape)

    for it in range(positions.shape[1]):
        qolo.pose.position = positions[:, it]
        temp_scan = reset_laserscan(qolo.get_allscan(), positions[:, it])

        _, _, relative_distances = qolo.get_relative_positions_and_dists(
            temp_scan, in_robot_frame=False
        )

        if any(relative_distances < 0):
            continue

        velocities_init[:, it] = dynamical_system.evaluate(positions[:, it])

        # my_avoider.update_laserscan(temp_scan)
        my_avoider.update_reference_direction(temp_scan, in_robot_frame=False)

        velocities_mod[:, it] = my_avoider.avoid(velocities_init[:, it])
        reference_dirs[:, it] = my_avoider.reference_direction

    # ax.quiver(
    #     positions[0, :].reshape(nx, ny),
    #     positions[1, :].reshape(nx, ny),
    #     reference_dirs[0, :].reshape(nx, ny),
    #     reference_dirs[1, :].reshape(nx, ny),
    #     color='red',
    # )
    plot_environment(
        ax, my_avoider, positions, velocities_mod, dynamical_system, x_lim, y_lim
    )


def visualization_analytic_data(
    qolo, dynamical_system, x_lim, y_lim, ax, n_sampels=10, **kwargs
):
    obstacle_avoider = FastObstacleAvoider(
        obstacle_environment=qolo.obstacle_environment, robot=qolo
    )
    qolo.obstacle_environment.update_reference_points()

    nx = ny = n_sampels
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    velocities_init = np.zeros(positions.shape)
    velocities_mod = np.zeros(positions.shape)
    reference_dirs = np.zeros(positions.shape)

    for it in range(positions.shape[1]):
        qolo.pose.position = positions[:, it]
        qolo._got_new_obstacles = True  #

        if is_interesecting(qolo.obstacle_environment, positions[:, it]):
            continue

        obstacle_avoider.update_reference_direction(in_robot_frame=False)

        # main_avoider.update_reference_direction(position=positions[:, it])
        # initial_vel = initial_dynamics.evaluate(position=positions[:, it])

        velocities_init[:, it] = dynamical_system.evaluate(positions[:, it])
        velocities_mod[:, it] = obstacle_avoider.avoid(velocities_init[:, it])
        reference_dirs[:, it] = obstacle_avoider.reference_direction

    plot_environment(
        ax,
        obstacle_avoider,
        positions,
        velocities_mod,
        dynamical_system,
        x_lim=x_lim,
        y_lim=y_lim,
        **kwargs
    )
    pass


def visualization_mixed(
    qolo, dynamical_system, x_lim, y_lim, ax, n_sampels=10, show_quiver=False
):
    """Draw the vectorfield mixed"""

    mixed_avoider = MixedEnvironmentAvoider(
        robot=qolo,
        scaling_obstacle_weight=50,
        scaling_laserscan_weight=1,
    )

    nx = ny = n_sampels
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    velocities_init = np.zeros(positions.shape)
    velocities_mod = np.zeros(positions.shape)
    reference_dirs = np.zeros(positions.shape)

    for it in range(positions.shape[1]):
        qolo.pose.position = positions[:, it]
        qolo._got_new_obstacles = True  #

        temp_scan = reset_laserscan(qolo.get_allscan(), positions[:, it])

        _, _, relative_distances = qolo.get_relative_positions_and_dists(
            temp_scan, in_robot_frame=False
        )

        if any(relative_distances < 0):
            continue

        if is_interesecting(qolo.obstacle_environment, positions[:, it]):
            continue

        mixed_avoider.update_laserscan(temp_scan)
        mixed_avoider.update_reference_direction(in_robot_frame=False)

        velocities_init[:, it] = dynamical_system.evaluate(positions[:, it])
        velocities_mod[:, it] = mixed_avoider.avoid(velocities_init[:, it])
        reference_dirs[:, it] = mixed_avoider.reference_direction

    plot_environment(
        ax=ax,
        my_avoider=mixed_avoider,
        positions=positions,
        velocities_mod=velocities_mod,
        dynamical_system=dynamical_system,
        x_lim=x_lim,
        y_lim=y_lim,
        show_quiver=show_quiver,
    )
    # ax,
    # my_avoider,
    # positions=None,
    # velocities_mod=None,
    # dynamical_system=None,
    # x_lim=None,
    # y_lim=None,
    # draw_velocity_arrow=False,
    # show_quiver=False,


def get_random_start_and_stop(x_lim=[-7, 7], y_vals=[-6, 6], y_lim=[-7, 9]):

    x_vals = np.random.rand(2) * (x_lim[1] - x_lim[0]) + x_lim[0]
    y_vals = np.random.rand(2) * (y_lim[1] - y_lim[0]) + y_lim[0]

    # start_position = np.array([x_vals[0], y_vals[0]])
    start_position = np.array([-5, y_vals[0]])

    # attractor_position = np.array([x_vals[1], y_vals[1]])

    attractor_position = np.array([11.0, 8.0])

    return start_position, attractor_position


def time_integration_fast(
    avoider,
    dynamical_system,
    it_max=350,
    dt=0.1,
    conv_margin=1e-1,
    conv_margin_ds=1e-2,
    ax=None,
    lasercan=None,
):
    """Returns the it-number at which it converged.

    Return Values:
    >= 0: the iteration number at which it converged
    -1 : no convergence in sufficient time
    -2 : Stuck
    -3 : Colliding / stuck (within obstacle)
    Return '"""
    it = 0

    temp_pos = np.zeros((2, it_max + 1))
    temp_pos[:, 0] = avoider.robot.pose.position

    final_it = -1
    while it < it_max:
        it += 1

        # Do such that the mixed avoider updates everything
        avoider.robot._got_new_obstacles = True
        if hasattr(avoider, "update_laserscan"):
            avoider.update_laserscan(lasercan, in_robot_frame=False)

        initial_ds = dynamical_system.evaluate(avoider.robot.pose.position)
        avoider.update_reference_direction()
        modulated_ds = avoider.avoid(initial_ds)

        avoider.robot.pose.position = avoider.robot.pose.position + modulated_ds * dt

        temp_pos[:, it] = avoider.robot.pose.position

        if (
            LA.norm(avoider.robot.pose.position - dynamical_system.attractor_position)
            < conv_margin
        ):
            final_it = it
            break

        _, _, relative_distances = qolo.get_relative_positions_and_dists(
            avoider.robot.get_allscan(), in_robot_frame=False
        )

        if LA.norm(modulated_ds) < conv_margin_ds:
            # Stuck colliding
            final_it = -2
            break

        if any(relative_distances < 0):
            # Stuck / colliding
            final_it = -3
            break

        if hasattr(avoider, "obstacle_environment") and is_interesecting(
            avoider.obstacle_environment, avoider.robot.pose.position
        ):
            # Stuck / colliding
            final_it = -3
            break

        # if ax is not None:
        # ax.plot(temp_pos[0, :it-1], temp_pos[1, :it-1], 'b')
        # breakpoint()

    if ax is not None:
        ax.plot(temp_pos[0, :it], temp_pos[1, :it])

    if final_it is None:
        final_it = -1

    # Not converged
    return final_it


def compare_numerically_algorithms(qolo):
    np.random.seed(0)

    n_runs = 10
    n_algos = 2

    run_results = np.zeros((n_algos, n_runs))

    dynamical_system = LinearSystem(
        attractor_position=np.array([11.0, 8.0]),
        maximum_velocity=0.8,
    )

    temp_scan = qolo.get_allscan()

    sample_avoider = SampledAvoider(robot=qolo)
    sample_avoider.update_laserscan(temp_scan)

    mixed_avoider = MixedEnvironmentAvoider(
        qolo,
        # scaling_laserscan_weight=0.5,
        scaling_obstacle_weight=50.0,
    )
    qolo.obstacle_environment.update_reference_points()

    obstacle_avoider = FastObstacleAvoider(
        obstacle_environment=qolo.obstacle_environment,
        robot=qolo,
    )

    mixed_avoider.update_laserscan(temp_scan)  # Update once before plotting

    # Has to be done each run (!)
    # qolo._has_new_obstacles = True
    # mixed_avoider.update_laserscan() # Update once before plotting

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    x_lim = [-7, 14]
    y_lim = [-7, 9]

    plot_environment(
        ax=axs[0],
        my_avoider=sample_avoider,
        dynamical_system=dynamical_system,
        x_lim=x_lim,
        y_lim=y_lim,
    )

    plot_environment(
        ax=axs[1],
        my_avoider=mixed_avoider,
        dynamical_system=dynamical_system,
        x_lim=x_lim,
        y_lim=y_lim,
    )

    for ii in range(n_runs):
        start_position, attractor_position = get_random_start_and_stop()

        # Do for the samples
        dynamical_system.attractor_position = attractor_position
        sample_avoider.robot.pose.position = start_position

        run_results[0, ii] = time_integration_fast(
            sample_avoider, dynamical_system, ax=axs[0], lasercan=np.copy(temp_scan)
        )

        # Do for the mixed environment
        dynamical_system.attractor_position = attractor_position
        sample_avoider.robot.pose.position = start_position
        qolo._got_new_obstacles = True

        # mixed_avoider.update_laserscan()
        run_results[1, ii] = time_integration_fast(
            mixed_avoider, dynamical_system, ax=axs[1], lasercan=np.copy(temp_scan)
        )

    print("Sampled Avoider Results")
    print(run_results)

    return run_results


def compare_two_vectorfields(qolo, save_figure=False, n_sampels=100):
    dynamical_system = LinearSystem(
        attractor_position=np.array([11.0, 8.0]),
        maximum_velocity=0.8,
    )

    x_lim = [-7, 14]
    y_lim = [-7, 9]

    fig, ax = plt.subplots(figsize=(6, 5))
    visualization_mixed(
        ax=ax,
        qolo=qolo,
        dynamical_system=dynamical_system,
        x_lim=x_lim,
        y_lim=y_lim,
        n_sampels=n_sampels,
        show_quiver=True,
    )

    if True:
        return

    if save_figure:
        figure_name = "mixed_environment_scan"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 5))
    visualization_sampledata(
        ax, qolo, dynamical_system, x_lim, y_lim, n_sampels=n_sampels
    )

    if save_figure:
        figure_name = "mixed_environment_both"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

    fig, ax = plt.subplots(figsize=(6, 5))
    visualization_analytic_data(
        qolo,
        dynamical_system,
        n_sampels=n_sampels,
        ax=ax,
        x_lim=x_lim,
        y_lim=y_lim,
        show_quiver=True,
    )

    if save_figure:
        figure_name = "mixed_environment_obstacle_only"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

    # Could be done...
    # visualization_obstacle(qolo, dynamical_system)


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # run_vectorfield_mixed()

    # do_the_import = True
    do_the_import = False
    if not "qolo" in vars() or not "qolo" in globals() or do_the_import:
        qolo = QoloRobot(
            pose=ObjectPose(position=[0.7, -0.7], orientation=30 * np.pi / 180)
        )

        import_first_scan_and_crowd(
            robot=qolo,
            bag_name="2021-12-03-18-21-29.bag",
            bag_dir="/home/lukas/Code/data_qolo/outdoor_recording/",
            start_time=None,
        )

    # compare_two_vectorfields(qolo=qolo, save_figure=False, n_sampels=50)
    compare_numerically_algorithms(qolo=qolo)
