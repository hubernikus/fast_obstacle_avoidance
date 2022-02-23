""" Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2021-12-14
# Email: lukas.huber@epfl.ch

from timeit import default_timer as timer

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import shapely

from vartools.states import ObjectPose
from vartools.dynamical_systems import ConstantValue, LinearSystem

from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer
from fast_obstacle_avoidance.sampling_container import visualize_obstacles


def explore_specific_point():
    qolo = QoloRobot(pose=ObjectPose(position=[7.0, -2.0], orientation=0))

    # dynamical_system = ConstantValue(velocity=[0, 1])
    dynamical_system = LinearSystem(
        attractor_position=np.array([0, 3]), maximum_velocity=1.0
    )

    fast_avoider = SampledAvoider(
        robot=qolo,
        # evaluate_normal=False,
        evaluate_normal=True,
        weight_max_norm=1e4,
        weight_factor=2,
        weight_power=1.0,
    )

    main_environment = ShapelySamplingContainer(n_samples=100)

    main_environment.add_obstacle(shapely.geometry.box(-5, -1, 2, 1))

    fast_avoider = SampledAvoider(
        robot=qolo,
        # evaluate_normal=False,
        evaluate_normal=True,
        weight_max_norm=1e5,
        weight_factor=3,
        weight_power=2.0,
    )

    dynamical_system = ConstantValue(velocity=[0, 1])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    data_points = main_environment.get_surface_points(
        center_position=qolo.pose.position,
    )

    ax.plot(data_points[0, :], data_points[1, :], "k.")

    ax.plot(qolo.pose.position[0], qolo.pose.position[1], "ko")

    visualize_obstacles(main_environment, ax=ax)

    eval_pos = qolo.pose.position

    fast_avoider.debug_mode = True
    fast_avoider.update_reference_direction(data_points, in_robot_frame=False)

    arrow_width = 0.01
    arrow_head_width = 0.1

    ax.arrow(
        eval_pos[0],
        eval_pos[1],
        fast_avoider.reference_direction[0],
        fast_avoider.reference_direction[1],
        color="blue",
        width=arrow_width,
        head_width=arrow_head_width,
    )

    ax.quiver(
        data_points[0, :],
        data_points[1, :],
        fast_avoider.ref_dirs[0, :],
        fast_avoider.ref_dirs[1, :],
        scale=10,
        color="blue",
        # width=arrow_width,
        alpha=0.8,
    )

    plot_normal_direction = False
    if plot_normal_direction:
        ax.arrow(
            eval_pos[0],
            eval_pos[1],
            fast_avoider.normal_direction[0],
            fast_avoider.normal_direction[1],
            color="red",
            width=arrow_width,
            head_width=arrow_head_width,
        )

        ax.quiver(
            data_points[0, :],
            data_points[1, :],
            fast_avoider.normal_dirs[0, :],
            fast_avoider.normal_dirs[1, :],
            scale=10,
            color="red",
            # width=arrow_width,
            alpha=0.8,
        )

    ax.set_aspect("equal")
    ax.grid(True)


def normal_offset_around_edge(main_environment=None, x_lim=None, y_lim=None):
    qolo = QoloRobot(pose=ObjectPose(position=[0.0, -3.4], orientation=0))

    # dynamical_system = ConstantValue(velocity=[0, 1])
    # dynamical_system = LinearSystem(
    # attractor_position=np.array([-1, 3]), maximum_velocity=1.0)

    if main_environment is None:
        main_environment = ShapelySamplingContainer(n_samples=100)

        main_environment.add_obstacle(shapely.geometry.box(-5, -1, 2, 1))

        # circle =   # type(circle)=polygon
        # ellipse = shapely.affinity.scale(shapely.geometry.Point(-1.5, 0).buffer(1), 5, 1.5)
        # ellipse = shapely.affinity.rotate(ellipse, -30)
        # main_environment.add_obstacle(
        # ellipse
        # )

    eval_pos = np.array([0, 0])
    qolo = QoloRobot(pose=ObjectPose(position=eval_pos, orientation=0))

    fast_avoider = SampledAvoider(
        robot=qolo,
        evaluate_normal=False,
        # evaluate_normal=True,
        weight_max_norm=1e4,
        weight_factor=2,
        weight_power=2.0,
    )

    # dynamical_system = ConstantValue(velocity=[0, 1])

    dynamical_system = LinearSystem(
        attractor_position=np.array([1, 3]), maximum_velocity=1.0
    )

    if x_lim is None:
        x_lim = [-1, 8]
    if y_lim is None:
        y_lim = [-4, 3]

    nx = ny = 30
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    velocities_init = np.zeros(positions.shape)
    velocities_mod = np.zeros(positions.shape)

    reference_dirs = np.zeros(positions.shape)
    norm_dirs = np.zeros(positions.shape)

    for it in range(positions.shape[1]):
        if main_environment.is_inside(positions[:, it]):
            continue

        qolo.pose.position = positions[:, it]

        data_points = main_environment.get_surface_points(
            center_position=positions[:, it],
        )

        _, _, relative_distances = qolo.get_relative_positions_and_dists(
            data_points, in_robot_frame=False
        )

        if any(relative_distances < 0):
            continue
        try:
            fast_avoider.update_reference_direction(data_points, in_robot_frame=False)

            velocities_init[:, it] = dynamical_system.evaluate(positions[:, it])
            velocities_mod[:, it] = fast_avoider.avoid(velocities_init[:, it])
        except:
            # breakpoint()
            # pass
            raise

        # Reference and normal dir
        reference_dirs[:, it] = fast_avoider.reference_direction
        norm_dirs[:, it] = fast_avoider.normal_direction

    plot_normals = True
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

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(data_points[0, :], data_points[1, :], "k.")

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

    visualize_obstacles(main_environment, ax=ax)

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_aspect("equal")
    ax.grid(True)


def execute_normal_offset_with_circle():
    main_environment = ShapelySamplingContainer(n_samples=100)

    # main_environment.add_obstacle(
    # shapely.geometry.box(-5, -1, 2, 1)
    # )

    # circle =   # type(circle)=polygon
    ellipse = shapely.affinity.scale(shapely.geometry.Point(1, 0).buffer(1), 1, 1)

    # ellipse = shapely.affinity.rotate(ellipse, -30)
    main_environment.add_obstacle(ellipse)

    normal_offset_around_edge(
        main_environment=main_environment, x_lim=[-3, 3], y_lim=[-3, 3]
    )


def test_surface_points():
    main_environment = ShapelySamplingContainer()
    main_environment.add_obstacle(shapely.geometry.box(0, 2, -1, 1))

    ax = visualize_obstacles(main_environment)

    center_position = np.array(np.array([0, 3]))

    sample_list = main_environment.get_surface_points(
        center_position=center_position, n_samples=20
    )

    ax.plot(sample_list[0, :], sample_list[1, :], "o", color="k")
    ax.plot(center_position[0], center_position[1], "o", color="r")

    ax.set_aspect("equal")
    ax.grid(True)

    x_lim = [-2, 5]
    y_lim = [-1, 6]

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")

    # explore_specific_point()
    # normal_offset_around_edge()
    execute_normal_offset_with_circle()

    # test_surface_points()
