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

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.visualization import plot_obstacles

# from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider
from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer
from fast_obstacle_avoidance.sampling_container import visualize_obstacles

from fast_obstacle_avoidance.visualization import FastObstacleAnimator
from fast_obstacle_avoidance.visualization import (
    static_visualization_of_sample_avoidance_obstacle,
)


def explore_specific_point_obstacle(
    robot=None,
    dynamical_system=None,
    fast_avoider=None,
    main_environment=None,
    x_lim=None,
    y_lim=None,
    draw_robot=False,
    show_ticks=False,
    ax=None,
    draw_velocity=False,
):
    if robot is None:
        robot = QoloRobot(pose=ObjectPose(position=[7.0, -2.0], orientation=0))

    eval_pos = robot.pose.position

    fast_avoider.debug_mode = True
    fast_avoider.update_reference_direction(position=robot.pose.position)

    initial_velocity = dynamical_system.evaluate(robot.pose.position)
    modulated_velocity = fast_avoider.avoid(initial_velocity)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

    # visualize_obstacles(main_environment, ax=ax)
    plot_obstacles(ax=ax, obstacle_container=main_environment, x_lim=x_lim, y_lim=y_lim)

    surface_points = []
    is_collision_free = np.ones(len(main_environment), dtype=bool)
    for ii, obs in enumerate(main_environment):
        pos_surface = obs.get_local_radius_point(
            robot.pose.position - obs.center_position, in_global_frame=True
        )

        for jj, obs_temp in enumerate(main_environment):
            if ii == jj:
                continue

            if obs_temp.get_gamma(pos_surface, in_global_frame=True) < 1:
                is_collision_free[ii] = False
                break

        if is_collision_free[ii]:
            surface_points.append(pos_surface)

    # Make sure only to visualize one for mutli-obstacle-obstacles
    ref_dirs = fast_avoider.ref_dirs[:, is_collision_free]
    ref_dirs = ref_dirs * 0.8

    tang_dirs = fast_avoider.tang_dirs[:, is_collision_free]
    tang_dirs = tang_dirs * 0.8

    surface_points = np.array(surface_points).T

    ax.plot(
        surface_points[0, :],
        surface_points[1, :],
        "o",
        color="black",
        markersize=11,
        zorder=3,
    )
    ax.plot(
        robot.pose.position[0],
        robot.pose.position[1],
        "o",
        color="black",
        markersize=13,
    )

    if draw_velocity:
        arrow_scale = 0.5
        arrow_width = 0.07
        arrow_headwith = 0.4
        margin_velocity_plot = 1e-3

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

    arrow_width = 0.04
    arrow_head_width = 0.2

    ax.arrow(
        eval_pos[0],
        eval_pos[1],
        fast_avoider.reference_direction[0],
        fast_avoider.reference_direction[1],
        color="#9b1503",
        width=arrow_width,
        head_width=arrow_head_width,
    )

    ax.quiver(
        surface_points[0, :],
        surface_points[1, :],
        # fast_avoider.ref_dirs[0, :],
        # fast_avoider.ref_dirs[1, :],
        ref_dirs[0, :],
        ref_dirs[1, :],
        scale=10,
        color="#9b1503",
        # color="blue",
        # width=arrow_width,
        alpha=1.0,
        label="Reference direction",
    )

    ax.arrow(
        eval_pos[0],
        eval_pos[1],
        fast_avoider.tangent_direction[0],
        fast_avoider.tangent_direction[1],
        color="#083e00ff",
        width=arrow_width,
        head_width=arrow_head_width,
    )

    ax.quiver(
        surface_points[0, :],
        surface_points[1, :],
        # fast_avoider.tang_dirs[0, :],
        # fast_avoider.tang_dirs[1, :],
        tang_dirs[0, :],
        tang_dirs[1, :],
        scale=10,
        # color="red",
        color="#083e00ff",
        # width=arrow_width,
        alpha=1.0,
        label="Tangent direction",
        zorder=2,
    )

    ax.legend(loc="upper left", fontsize=12)

    robot.plot2D(ax=ax)

    ax.set_aspect("equal")

    if not show_ticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    # ax.grid(True)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    return ax


def vectorfield_with_many_obstacles(create_animation=False, save_figure=False):
    start_point = np.array([9, 6])
    x_lim = [0, 18]
    y_lim = [0, 8]

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([17, 1.0]), maximum_velocity=1.0
    )

    robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))

    robot.control_point = [0, 0]
    robot.control_radius = 0.6

    main_environment = GradientContainer()
    main_environment.append(
        Ellipse(
            center_position=np.array([2, 6]),
            orientation=0,
            axes_length=np.array([0.4, 0.8]),
            margin_absolut=robot.control_radius,
        )
    )

    main_environment.append(
        Ellipse(
            center_position=np.array([5.5, 2]),
            orientation=-40 * np.pi / 180,
            axes_length=np.array([0.3, 1.3]),
            margin_absolut=robot.control_radius,
            angular_velocity=10 * np.pi / 180,
        )
    )

    main_environment.append(
        Ellipse(
            center_position=np.array([14, 7]),
            orientation=-20 * np.pi / 180,
            axes_length=np.array([0.3, 0.9]),
            margin_absolut=robot.control_radius,
            linear_velocity=np.array([-0.7, -0.2]),
        )
    )

    main_environment.append(
        Ellipse(
            center_position=np.array([13, 2]),
            orientation=30 * np.pi / 180,
            axes_length=np.array([0.3, 2.1]),
            margin_absolut=robot.control_radius,
        )
    )
    # Crossbone
    main_environment.append(copy.deepcopy(main_environment[-1]))
    main_environment[-1].orientation = (
        main_environment[-1].orientation + 90 * np.pi / 180
    )

    fast_avoider = FastObstacleAvoider(
        obstacle_environment=main_environment,
        robot=robot,
        weight_max_norm=1e5,
        weight_factor=5,
        weight_power=2.5,
    )

    if create_animation:
        plt.close("all")
        simu_environment = copy.deepcopy(main_environment)
        simu_environment.n_samples = 100

        simu_bot = copy.deepcopy(robot)
        simu_bot.pose.position = np.array([1.8, 4.1])

        my_animator = FastObstacleAnimator(
            it_max=400,
            dt_simulation=0.05,
        )

        fast_avoider.robot = simu_bot

        my_animator.setup(
            robot=simu_bot,
            initial_dynamics=initial_dynamics,
            avoider=fast_avoider,
            environment=simu_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            show_reference=True,
        )

        my_animator.run(save_animation=save_figure)
        return

    # Plot the vectorfield around the robot
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    explore_specific_point_obstacle(
        robot=robot,
        dynamical_system=initial_dynamics,
        fast_avoider=fast_avoider,
        main_environment=main_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        draw_velocity=True,
    )

    static_visualization_of_sample_avoidance_obstacle(
        robot=robot,
        n_resolution=100,
        dynamical_system=initial_dynamics,
        fast_avoider=fast_avoider,
        plot_initial_robot=True,
        main_environment=main_environment,
        show_ticks=True,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
    )

    if save_figure:
        figure_name = "multiple_avoiding_obstacles_analytic"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    plt.ion()
    # plt.close("all")

    # test_multi_obstacles()
    # vectorfield_with_many_obstacles(save_figure=False, create_animation=False)

    # test_various_surface_points()
