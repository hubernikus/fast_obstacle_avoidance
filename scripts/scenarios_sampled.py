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

from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer
from fast_obstacle_avoidance.sampling_container import visualize_obstacles

from fast_obstacle_avoidance.visualization import LaserscanAnimator
from fast_obstacle_avoidance.visualization import (
    static_visualization_of_sample_avoidance,
)


def get_random_position(x_lim, y_lim=None):
    dimension = 2

    pos = np.random.rand(2)
    pos[0] = pos[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    if y_lim is None:
        return pos[0]

    pos[1] = pos[1] * (y_lim[1] - y_lim[0]) + y_lim[0]

    return pos


def explore_specific_point(
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

    if fast_avoider is None:
        breakpoint()
        # fast_avoider = SampledAvoider(
        #     robot=robot,
        #     # evaluate_normal=False,
        #     evaluate_normal=True,
        #     weight_max_norm=1e6,
        #     weight_factor=1,
        #     weight_power=1.0,
        #     )

    if main_environment is None:
        main_environment = ShapelySamplingContainer(n_samples=100)
        main_environment.add_obstacle(shapely.geometry.box(-5, -1, 2, 1))

    if dynamical_system is None:
        dynamical_system = ConstantValue(velocity=[0, 1])

    data_points = main_environment.get_surface_points(
        center_position=robot.pose.position,
    )

    eval_pos = robot.pose.position

    fast_avoider.debug_mode = True
    fast_avoider.update_reference_direction(data_points, in_robot_frame=False)

    initial_velocity = dynamical_system.evaluate(robot.pose.position)
    modulated_velocity = fast_avoider.avoid(initial_velocity)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

    visualize_obstacles(main_environment, ax=ax)

    ax.plot(data_points[0, :], data_points[1, :], "o", color="black", markersize=11)
    ax.plot(
        robot.pose.position[0],
        robot.pose.position[1],
        "o",
        color="black",
        markersize=13,
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
        data_points[0, :],
        data_points[1, :],
        fast_avoider.ref_dirs[0, :],
        fast_avoider.ref_dirs[1, :],
        scale=10,
        color="#9b1503",
        # color="blue",
        # width=arrow_width,
        alpha=1.0,
        label="Reference direction",
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

        ax.legend(loc="upper left", fontsize=12)

    plot_normal_direction = False
    if plot_normal_direction:
        self.ax.arrow(
            eval_pos[0],
            eval_pos[1],
            fast_avoider.normal_direction[0],
            fast_avoider.normal_direction[1],
            color="red",
            width=arrow_width,
            head_width=arrow_head_width,
        )

        self.ax.quiver(
            data_points[0, :],
            data_points[1, :],
            fast_avoider.normal_dirs[0, :],
            fast_avoider.normal_dirs[1, :],
            scale=10,
            color="red",
            # width=arrow_width,
            alpha=0.8,
        )

    robot.plot2D(ax=ax)

    ax.set_aspect("equal")

    if not show_ticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    # ax.grid(True)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    return ax


def execute_avoidance_with_obstacle(save_figure=False):
    """Visualizes
    (1) the repulsion from a specific point
    (2) the resulting vector-field."""

    start_point = np.array([-2.5, 3])
    x_lim = [-4, 4]
    y_lim = [-1.5, 5.6]

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([3.5, 1.3]), maximum_velocity=1.0
    )

    main_environment = ShapelySamplingContainer(n_samples=50)
    # main_environment.add_obstacle(shapely.geometry.box(-5, -1, 2, 1))

    # main_environment = ShapelySamplingContainer(n_samples=100)

    # main_environment.add_obstacle(shapely.geometry.box(-5, -1, 2, 1))
    # circle =   # type(circle)=polygon

    ellipse = shapely.affinity.scale(
        shapely.geometry.Point(0.5, -0.5).buffer(1), 2.0, 1.5
    )
    ellipse = shapely.affinity.rotate(ellipse, 90)

    main_environment.add_obstacle(ellipse)
    robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.6

    fast_avoider = SampledAvoider(
        robot=robot,
        weight_max_norm=1e8,
        weight_factor=4,
        weight_power=2.0,
    )

    # Do the animation, too:
    create_animation = True
    if create_animation:
        simu_environment = copy.deepcopy(main_environment)
        simu_environment.n_samples = 100

        simu_bot = copy.deepcopy(robot)
        simu_bot.pose.position = np.array([-2.5, 1])

        my_animator = LaserscanAnimator(
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
        )

        my_animator.run(save_animation=save_figure)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    explore_specific_point(
        robot=robot,
        dynamical_system=initial_dynamics,
        fast_avoider=fast_avoider,
        main_environment=main_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
    )

    # Draw initial dynamical system
    plot_dynamical_system_streamplot(
        dynamical_system=initial_dynamics,
        fig_ax_handle=(fig, ax),
        x_lim=x_lim,
        y_lim=y_lim,
        color="#837cfd",
        zorder=-10,
    )

    ax.plot(
        initial_dynamics.attractor_position[0],
        initial_dynamics.attractor_position[1],
        "k*",
        linewidth=18.0,
        markersize=18,
        zorder=5,
    )

    if save_figure:
        figure_name = "visualization_reference_vectors"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

    # Plot the vectorfield around the robot
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    static_visualization_of_sample_avoidance(
        robot=robot,
        dynamical_system=initial_dynamics,
        fast_avoider=fast_avoider,
        plot_initial_robot=True,
        main_environment=main_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
    )

    if save_figure:
        figure_name = "visualize_avoiding_field"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def vectorfield_with_many_obstacles(save_figure=False):
    start_point = np.array([-1, 1])
    x_lim = [-8, 4]
    y_lim = [-1.0, 4.6]

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([3.5, 1.3]), maximum_velocity=1.0
    )

    main_environment = ShapelySamplingContainer(n_samples=50)

    # main_environment = ShapelySamplingContainer(n_samples=100)

    # main_environment.add_obstacle(shapely.geometry.box(-5, -1, 2, 1))
    # circle =   # type(circle)=polygon

    # Ellipse
    ellipse = shapely.affinity.scale(
        shapely.geometry.Point(-3, 3.5).buffer(1), 1.0, 0.8
    )
    ellipse = shapely.affinity.rotate(ellipse, 50)
    main_environment.add_obstacle(ellipse)

    # Ellipse
    ellipse = shapely.affinity.scale(shapely.geometry.Point(2, 0).buffer(1), 1.0, 0.8)
    ellipse = shapely.affinity.rotate(ellipse, -20)
    main_environment.add_obstacle(ellipse)

    # Box
    main_environment.add_obstacle(shapely.geometry.box(-6, -1, -5, 1.5))

    # Second Box
    main_environment.add_obstacle(shapely.geometry.box(0, 4, 1, 8))

    robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.6

    fast_avoider = SampledAvoider(
        robot=robot,
        weight_max_norm=1e6,
        weight_factor=1,
        weight_power=1.0,
    )

    create_animation = True
    if create_animation:
        simu_environment = copy.deepcopy(main_environment)
        simu_environment.n_samples = 100

        simu_bot = copy.deepcopy(robot)
        simu_bot.pose.position = np.array([-7.5, 0.8])

        my_animator = LaserscanAnimator(
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

    main_environment.n_samples = 40
    explore_specific_point(
        robot=robot,
        dynamical_system=initial_dynamics,
        fast_avoider=fast_avoider,
        main_environment=main_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        draw_velocity=True,
    )

    main_environment.n_samples = 50
    static_visualization_of_sample_avoidance(
        robot=robot,
        n_resolution=50,
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
        figure_name = "multiple_avoiding_obstacles"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def test_multi_obstacles():
    start_point = np.array([-7.5, 0.8])
    # start_point = np.array([0.5, 1.4])
    x_lim = [-8, 4]
    y_lim = [-1.0, 4.6]

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([3.5, 1.3]), maximum_velocity=1.0
    )

    main_environment = ShapelySamplingContainer(n_samples=50)

    # main_environment = ShapelySamplingContainer(n_samples=100)

    # main_environment.add_obstacle(shapely.geometry.box(-5, -1, 2, 1))
    # circle =   # type(circle)=polygon

    # Ellipse
    ellipse = shapely.affinity.scale(
        shapely.geometry.Point(-3, 3.5).buffer(1), 1.0, 0.8
    )
    ellipse = shapely.affinity.rotate(ellipse, 50)
    main_environment.add_obstacle(ellipse)

    ellipse = shapely.affinity.scale(shapely.geometry.Point(2, 0).buffer(1), 1.0, 0.8)
    ellipse = shapely.affinity.rotate(ellipse, -20)
    main_environment.add_obstacle(ellipse)

    # Box
    main_environment.add_obstacle(shapely.geometry.box(-6, -1, -5, 1.5))

    # Second Box
    main_environment.add_obstacle(shapely.geometry.box(0, 4, 1, 8))

    robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.6

    fast_avoider = SampledAvoider(
        robot=robot,
        weight_max_norm=1e8,
        weight_factor=4,
        weight_power=2.0,
    )

    main_environment.n_samples = 100
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    explore_specific_point(
        robot=robot,
        dynamical_system=initial_dynamics,
        fast_avoider=fast_avoider,
        main_environment=main_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        draw_velocity=True,
    )


def multiple_random_cirlces():
    np.random.seed(8)

    human_radius = 0.7

    attractor_position = np.array([5.5, 4])

    # start_point = np.array([0.5, 1.4])
    x_lim = [-6, 6]
    y_lim = [-4.5, 4.5]

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=attractor_position, maximum_velocity=1.0
    )

    robot = QoloRobot(pose=ObjectPose(position=np.zeros([0, 0]), orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.6

    delta_dist = robot.control_radius + human_radius
    x_lim_obs = [x_lim[0] + delta_dist, attractor_position[1] - delta_dist]
    # y_lim_obs = [y_lim[0]+delta_dist , y_lim[1]-delta_dist]

    n_humans = 5

    main_environment = ShapelySamplingContainer(n_samples=50)
    for ii in range(n_humans):
        # Random ellipse
        position = get_random_position(x_lim=x_lim_obs, y_lim=y_lim)
        main_environment.create_sphere(position, radius=human_radius)

    fast_avoider = SampledAvoider(
        robot=robot,
        weight_max_norm=1e6,
        weight_factor=1,
        weight_power=1.0,
    )

    main_environment.n_samples = 100

    # fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    fig, axs = plt.subplots(1, 2, figsize=(16, 9))

    static_visualization_of_sample_avoidance(
        robot=robot,
        n_resolution=40,
        dynamical_system=initial_dynamics,
        fast_avoider=fast_avoider,
        plot_initial_robot=False,
        main_environment=main_environment,
        show_ticks=True,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=axs[0],
        plot_quiver=True,
        ax_ref=axs[1],
    )


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")

    # execute_avoidance_with_obstacle(save_figure=True)
    # test_multi_obstacles()
    # vectorfield_with_many_obstacles(save_figure=False)
    # vectorfield_with_many_obstacles(save_figure=False)

    # test_various_surface_points()

    multiple_random_cirlces()