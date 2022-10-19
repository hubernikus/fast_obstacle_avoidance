""" Script to create plots. """
# Author: Lukas Huber
# Created: 2021-02-22
# Email: lukas.huber@epfl.ch

import copy

from timeit import default_timer as timer

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import shapely

from vartools.states import ObjectPose
from vartools.dynamical_systems import ConstantValue, LinearSystem
from vartools.dynamical_systems import plot_dynamical_system_streamplot

from dynamic_obstacle_avoidance.containers import GradientContainer

from dynamic_obstacle_avoidance.obstacles import CircularObstacle
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider
from fast_obstacle_avoidance.obstacle_avoider import MixedEnvironmentAvoider

from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer
from fast_obstacle_avoidance.sampling_container import visualize_obstacles

from fast_obstacle_avoidance.visualization import LaserscanAnimator
from fast_obstacle_avoidance.visualization import FastObstacleAnimator
from fast_obstacle_avoidance.visualization import MixedObstacleAnimator


def cleanup_datapoints(data_points, robot):
    # Check that all are collision free
    ind_collision_free = np.ones(data_points.shape[1], dtype=bool)
    for pp in range(data_points.shape[1]):
        for obs in robot.obstacle_environment:
            if obs.get_gamma(data_points[:, pp], in_global_frame=True) <= 1:
                ind_collision_free[pp] = False
                break

    return data_points[:, ind_collision_free]


def static_visualization_of_sample_avoidance(
    main_environment,
    dynamical_system,
    fast_avoider=None,
    n_resolution=30,
    robot=None,
    show_ticks=False,
    plot_initial_robot=False,
    x_lim=None,
    y_lim=None,
    ax=None,
    plot_quiver=False,
    plot_ref_vectorfield=False,
    ax_ref=None,
    plot_velocities=False,
):
    """Visualization of sampled environment."""
    # circle =   # type(circle)=polygon
    # ellipse = shapely.affinity.scale(shapely.geometry.Point(-1.5, 0).buffer(1), 5, 1.5)
    # ellipse = shapely.affinity.rotate(ellipse, -30)
    # main_environment.add_obstacle(
    # ellipse
    # )

    if x_lim is None:
        x_lim = [-1, 8]

    if y_lim is None:
        y_lim = [-4, 3]

    eval_pos = np.array([0, 0])

    if robot is None:
        robot = QoloRobot(pose=ObjectPose(position=eval_pos, orientation=0))

    if plot_initial_robot:
        robot.plot2D(ax=ax)

        data_points = main_environment.get_surface_points(
            center_position=robot.pose.position,
        )

        fast_avoider.update_laserscan(data_points, in_robot_frame=False)

        # Store all
        initial_velocity = dynamical_system.evaluate(robot.pose.position)
        modulated_velocity = fast_avoider.avoid(initial_velocity)

        arrow_scale = 0.5
        arrow_width = 0.07
        arrow_headwith = 0.4
        margin_velocity_plot = 1e-3

        ax.plot(
            robot.pose.position[0],
            robot.pose.position[1],
            "o",
            color="black",
            markersize=13,
            zorder=5,
        )

        if plot_velocities:
            ax.arrow(
                robot.pose.position[0],
                robot.pose.position[1],
                arrow_scale * initial_velocity[0],
                arrow_scale * initial_velocity[1],
                width=arrow_width,
                head_width=arrow_headwith,
                # color="g",
                color="#008080",
                label="Initial velocity",
            )

            ax.arrow(
                robot.pose.position[0],
                robot.pose.position[1],
                arrow_scale * modulated_velocity[0],
                arrow_scale * modulated_velocity[1],
                width=arrow_width,
                head_width=arrow_headwith,
                # color="b",
                # color='#213970',
                color="#000080",
                label="Modulated velocity",
            )

        ax.arrow(
            robot.pose.position[0],
            robot.pose.position[1],
            fast_avoider.reference_direction[0],
            fast_avoider.reference_direction[1],
            color="#9b1503",
            width=arrow_width,
            head_width=arrow_headwith,
            label="Reference [summed]",
        )

    if fast_avoider is None:
        fast_avoider = SampledAvoider(
            robot=robot,
            evaluate_normal=False,
            # evaluate_normal=True,
            weight_max_norm=1e4,
            weight_factor=2,
            weight_power=2.0,
        )

    # dynamical_system = ConstantValue(velocity=[0, 1])

    if dynamical_system is None:
        dynamical_system = LinearSystem(
            attractor_position=np.array([1, 3]), maximum_velocity=1.0
        )

    nx = ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    velocities_init = np.zeros(positions.shape)
    velocities_mod = np.zeros(positions.shape)

    reference_dirs = np.zeros(positions.shape)
    norm_dirs = np.zeros(positions.shape)

    for it in range(positions.shape[1]):
        if main_environment.is_inside(
            positions[:, it], margin=robot.control_radius * 1.1
        ):
            # Put 1.1 margin for nicer plots
            continue

        robot.pose.position = positions[:, it]
        # robot.pose.position = np.array([-2.5, 3])

        data_points = main_environment.get_surface_points(
            center_position=positions[:, it],
        )

        _, _, relative_distances = robot.get_relative_positions_and_dists(
            data_points, in_robot_frame=False
        )

        if any(relative_distances < 0):
            continue

        # fast_avoider.update_reference_direction(data_points, in_robot_frame=False)
        fast_avoider.update_laserscan(data_points, in_robot_frame=False)

        velocities_init[:, it] = dynamical_system.evaluate(positions[:, it])
        velocities_mod[:, it] = fast_avoider.avoid(velocities_init[:, it])

        # Reference and normal dir
        if hasattr(fast_avoider, "reference_direction"):
            reference_dirs[:, it] = fast_avoider.reference_direction

        if hasattr(fast_avoider, "normal_direction"):
            norm_dirs[:, it] = fast_avoider.normal_direction

    plot_normals = False
    if plot_normals:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(data_points[0, :], data_points[1, :], "k.")

        ax.quiver(
            positions[0, :],
            positions[1, :],
            reference_dirs[0, :],
            reference_dirs[1, :],
            scale=30,
            color="black",
            # width=arrow_width,
            alpha=0.8,
        )

        ax.quiver(
            positions[0, :],
            positions[1, :],
            norm_dirs[0, :],
            norm_dirs[1, :],
            scale=30,
            color="r",
            # width=arrow_width,
            alpha=0.8,
        )

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        visualize_obstacles(main_environment, ax=ax)

        ax.set_aspect("equal")
        ax.grid(True)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

    # ax.plot(data_points[0, :], data_points[1, :], "k.")

    if plot_quiver:
        ax.quiver(
            positions[0, :],
            positions[1, :],
            velocities_mod[0, :],
            velocities_mod[1, :],
            angles="xy",
            scale_units="xy",
            # scale=scale_vel,
            # width=arrow_width,
            color="blue",
        )

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

    visualize_obstacles(main_environment, ax=ax)

    if hasattr(dynamical_system, "attractor_position"):
        ax.plot(
            dynamical_system.attractor_position[0],
            dynamical_system.attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if not show_ticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    ax.set_aspect("equal")
    # ax.grid(True)

    if plot_ref_vectorfield or ax_ref is not None:
        # _, ax_ref = plt.subplots(1, 1, figsize=(10, 6))
        if ax_ref is None:
            ax_ref = ax

        ax_ref.quiver(
            positions[0, :],
            positions[1, :],
            reference_dirs[0, :],
            reference_dirs[1, :],
            angles="xy",
            scale_units="xy",
            # scale=scale_vel,
            # width=arrow_width,
            color="red",
        )

        visualize_obstacles(main_environment, ax=ax_ref)

        ax_ref.set_xlim(x_lim)
        ax_ref.set_ylim(y_lim)

        if not show_ticks:
            ax_ref.axes.xaxis.set_visible(False)
            ax_ref.axes.yaxis.set_visible(False)

        ax_ref.set_aspect("equal")

    return ax


def static_visualization_of_sample_avoidance_obstacle(
    main_environment,
    dynamical_system,
    fast_avoider=None,
    n_resolution=30,
    robot=None,
    show_ticks=False,
    plot_initial_robot=False,
    x_lim=None,
    y_lim=None,
    ax=None,
    do_quiver=False,
    plot_ref_vectorfield=False,
):
    """Visualization of obstacle environment."""

    if plot_initial_robot:
        robot.plot2D(ax=ax)
        ax.plot(
            robot.pose.position[0],
            robot.pose.position[1],
            "o",
            color="black",
            markersize=13,
            zorder=5,
        )

    nx = ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    velocities_init = np.zeros(positions.shape)
    velocities_mod = np.zeros(positions.shape)

    reference_dirs = np.zeros(positions.shape)
    norm_dirs = np.zeros(positions.shape)

    for it in range(positions.shape[1]):

        is_inside_an_obstacle = False
        for obs in main_environment:
            if obs.get_gamma(positions[:, it], in_global_frame=True) < 1:
                is_inside_an_obstacle = True
                break

        if is_inside_an_obstacle:
            continue

        robot.pose.position = positions[:, it]

        fast_avoider.update_reference_direction(position=robot.pose.position)

        velocities_init[:, it] = dynamical_system.evaluate(positions[:, it])
        velocities_mod[:, it] = fast_avoider.avoid(velocities_init[:, it])

        # Reference and normal dir
        reference_dirs[:, it] = fast_avoider.reference_direction
        norm_dirs[:, it] = fast_avoider.normal_direction

    plot_normals = False
    if plot_normals:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(data_points[0, :], data_points[1, :], "k.")

        ax.quiver(
            positions[0, :],
            positions[1, :],
            reference_dirs[0, :],
            reference_dirs[1, :],
            scale=30,
            color="black",
            # width=arrow_width,
            alpha=0.8,
        )

        ax.quiver(
            positions[0, :],
            positions[1, :],
            norm_dirs[0, :],
            norm_dirs[1, :],
            scale=30,
            color="r",
            # width=arrow_width,
            alpha=0.8,
        )

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

        visualize_obstacles(main_environment, ax=ax)

        ax.set_aspect("equal")
        ax.grid(True)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

    # ax.plot(data_points[0, :], data_points[1, :], "k.")

    if do_quiver:
        ax.quiver(
            positions[0, :],
            positions[1, :],
            velocities_mod[0, :],
            velocities_mod[1, :],
            angles="xy",
            scale_units="xy",
            # scale=scale_vel,
            # width=arrow_width,
            color="blue",
        )

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

    # visualize_obstacles(main_environment, ax=ax)

    if hasattr(dynamical_system, "attractor_position"):
        ax.plot(
            dynamical_system.attractor_position[0],
            dynamical_system.attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if not show_ticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    ax.set_aspect("equal")
    # ax.grid(True)

    if plot_ref_vectorfield:
        # _, ax_ref = plt.subplots(1, 1, figsize=(10, 6))
        ax_ref = ax

        ax_ref.quiver(
            positions[0, :],
            positions[1, :],
            reference_dirs[0, :],
            reference_dirs[1, :],
            angles="xy",
            scale_units="xy",
            # scale=scale_vel,
            # width=arrow_width,
            color="red",
        )

        visualize_obstacles(main_environment, ax=ax_ref)

        ax_ref.set_xlim(x_lim)
        ax_ref.set_ylim(y_lim)

        if not show_ticks:
            ax_ref.axes.xaxis.set_visible(False)
            ax_ref.axes.yaxis.set_visible(False)

        ax_ref.set_aspect("equal")

    return ax


def static_visualization_of_sample_avoidance_mixed(
    sample_environment,
    dynamical_system,
    fast_avoider=None,
    n_resolution=30,
    robot=None,
    show_ticks=False,
    plot_initial_robot=False,
    x_lim=None,
    y_lim=None,
    ax=None,
    do_quiver=False,
    plot_ref_vectorfield=False,
    plot_norm_dirs=False,
    plot_velocities=False,
    ax_ref=None,
):
    """Visualization of mixed environment."""

    if plot_initial_robot:
        robot.plot2D(ax=ax)

        ax.plot(
            robot.pose.position[0],
            robot.pose.position[1],
            "o",
            color="black",
            markersize=13,
            zorder=5,
        )

        data_points = sample_environment.get_surface_points(
            center_position=robot.pose.position,
        )

        data_points = cleanup_datapoints(data_points, robot=robot)

        # fast_avoider.update_laserscan(data_points)
        # fast_avoider.update_reference_direction(in_robot_frame=False)
        fast_avoider.update_laserscan(data_points, in_robot_frame=False)

        # Store all
        initial_velocity = dynamical_system.evaluate(robot.pose.position)
        modulated_velocity = fast_avoider.avoid(initial_velocity)

        ax.plot(data_points[0, :], data_points[1, :], "o", color="k")

        arrow_scale = 0.5
        arrow_width = 0.07
        arrow_headwith = 0.4
        margin_velocity_plot = 1e-3

        if plot_velocities:
            ax.arrow(
                robot.pose.position[0],
                robot.pose.position[1],
                arrow_scale * initial_velocity[0],
                arrow_scale * initial_velocity[1],
                width=arrow_width,
                head_width=arrow_headwith,
                # color="g",
                color="#008080",
                label="Initial velocity",
            )

            ax.arrow(
                robot.pose.position[0],
                robot.pose.position[1],
                arrow_scale * modulated_velocity[0],
                arrow_scale * modulated_velocity[1],
                width=arrow_width,
                head_width=arrow_headwith,
                # color="b",
                # color='#213970',
                color="#000080",
                label="Modulated velocity",
            )

        ax.arrow(
            robot.pose.position[0],
            robot.pose.position[1],
            fast_avoider.reference_direction[0],
            fast_avoider.reference_direction[1],
            color="#9b1503",
            width=arrow_width,
            head_width=arrow_headwith,
            label="Reference [summed]",
        )

        ax.arrow(
            robot.pose.position[0],
            robot.pose.position[1],
            fast_avoider.obstacle_avoider.reference_direction[0],
            fast_avoider.obstacle_avoider.reference_direction[1],
            color="#CD7F32",
            width=arrow_width,
            head_width=arrow_headwith,
            label="Reference [analytic]",
            alpha=0.9,
        )

        ax.arrow(
            robot.pose.position[0],
            robot.pose.position[1],
            fast_avoider.lidar_avoider.reference_direction[0],
            fast_avoider.lidar_avoider.reference_direction[1],
            color="#3d3635",
            width=arrow_width,
            head_width=arrow_headwith,
            label="Reference [sampled]",
            alpha=0.9,
        )

        if plot_norm_dirs:
            ax.arrow(
                robot.pose.position[0],
                robot.pose.position[1],
                fast_avoider.normal_direction[0],
                fast_avoider.normal_direction[1],
                color="#702963",
                width=arrow_width,
                head_width=arrow_headwith,
                label="Normal",
                alpha=0.9,
            )

        ax.legend(loc="upper left", fontsize=12)

    nx = ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    velocities_init = np.zeros(positions.shape)
    velocities_mod = np.zeros(positions.shape)

    reference_dirs = np.zeros(positions.shape)
    norm_dirs = np.zeros(positions.shape)

    for it in range(positions.shape[1]):
        robot.pose.position = positions[:, it]

        is_inside_an_obstacle = False
        for obs in fast_avoider.obstacle_environment:
            if obs.get_gamma(robot.pose.position, in_global_frame=True) < 1:
                is_inside_an_obstacle = True
                break

        if is_inside_an_obstacle:
            continue

        if sample_environment.is_inside(
            position=robot.pose.position, margin=robot.control_radius
        ):
            continue

        # robot.pose.position = np.array([5.41, 5.99])
        # robot.pose.position = np.array([8.00, 2.55])
        data_points = sample_environment.get_surface_points(
            center_position=robot.pose.position,
        )

        data_points = cleanup_datapoints(data_points=data_points, robot=robot)

        # Update the obstacle environment
        fast_avoider.update_laserscan(data_points, in_robot_frame=False)

        velocities_init[:, it] = dynamical_system.evaluate(robot.pose.position)
        velocities_mod[:, it] = fast_avoider.avoid(velocities_init[:, it])

        # Reference and normal dir
        reference_dirs[:, it] = fast_avoider.reference_direction
        norm_dirs[:, it] = fast_avoider.normal_direction

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

    # ax.plot(data_points[0, :], data_points[1, :], "k.")

    if do_quiver:
        ax.quiver(
            positions[0, :],
            positions[1, :],
            velocities_mod[0, :],
            velocities_mod[1, :],
            angles="xy",
            scale_units="xy",
            # scale=scale_vel,
            # width=arrow_width,
            color="blue",
        )

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
            zorder=-2,
        )

    visualize_obstacles(sample_environment, ax=ax)

    plot_obstacles(
        ax=ax,
        obstacle_container=fast_avoider.obstacle_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        drawVelArrow=False,
    )

    if hasattr(dynamical_system, "attractor_position"):
        ax.plot(
            dynamical_system.attractor_position[0],
            dynamical_system.attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if not show_ticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    ax.set_aspect("equal")
    # ax.grid(True)

    if plot_ref_vectorfield or ax_ref is not None:
        # _, ax_ref = plt.subplots(1, 1, figsize=(10, 6))
        if ax_ref is None:
            ax_ref = ax

        ax_ref.quiver(
            positions[0, :],
            positions[1, :],
            reference_dirs[0, :],
            reference_dirs[1, :],
            angles="xy",
            scale_units="xy",
            # scale=scale_vel,
            # width=arrow_width,
            color="red",
            label="Reference directions",
        )

        if plot_norm_dirs:
            ax_ref.quiver(
                positions[0, :],
                positions[1, :],
                norm_dirs[0, :],
                norm_dirs[1, :],
                angles="xy",
                scale_units="xy",
                # scale=scale_vel,
                # width=arrow_width,
                color="green",
                label="Normal directions",
            )

        # visualize_obstacles(main_environment, ax=ax_ref)

        ax_ref.set_xlim(x_lim)
        ax_ref.set_ylim(y_lim)

        if not show_ticks:
            ax_ref.axes.xaxis.set_visible(False)
            ax_ref.axes.yaxis.set_visible(False)

        ax_ref.set_aspect("equal")

    return ax
