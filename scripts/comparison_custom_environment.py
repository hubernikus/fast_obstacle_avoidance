""" Script to create plots. """
# Author: Lukas Huber
# Created: 2021-03-02
# Email: lukas.huber@epfl.ch

import copy

from timeit import default_timer as timer

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import shapely

# from shapely import affinity

from vartools.states import ObjectPose
from vartools.dynamical_systems import LinearSystem

# from dynamic_obstacle_avoidance.obstacles import Ellipse, Cuboid
from dynamic_obstacle_avoidance.obstacles.ellipse_xd import EllipseWithAxes
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd

from dynamic_obstacle_avoidance.obstacles import CircularObstacle

# from dynamic_obstacle_avoidance.containers import SphereContainer
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer

from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider
from fast_obstacle_avoidance.obstacle_avoider import MixedEnvironmentAvoider

from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.visualization import MixedObstacleAnimator
from fast_obstacle_avoidance.visualization import FastObstacleAnimator
from fast_obstacle_avoidance.visualization import LaserscanAnimator

from fast_obstacle_avoidance.visualization.integration_plot import (
    visualization_mixed_environment_with_multiple_integration,
)

from fast_obstacle_avoidance.visualization import (
    static_visualization_of_sample_avoidance_mixed,
)


def get_random_position_orientation_and_axes(x_lim, y_lim, axes_range):
    dimension = 2

    position = np.random.rand(dimension)
    position[0] = position[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    position[1] = position[1] * (y_lim[1] - y_lim[0]) + y_lim[0]

    orientation_deg = np.random.rand(1) * 360

    axes_length = np.random.rand(2) * (axes_range[1] - axes_range[0]) + axes_range[0]

    return position, orientation_deg, axes_length


def get_random_position(x_lim, y_lim=None):
    dimension = 2

    pos = np.random.rand(2)
    pos[0] = pos[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    if y_lim is None:
        return pos[0]

    pos[1] = pos[1] * (y_lim[1] - y_lim[0]) + y_lim[0]

    return pos


def visualize_vectorfield_mixed():
    """Visualize vectorfield of mixed environment."""
    np.random.seed(1)

    (
        robot,
        initial_dynamics,
        main_environment,
        obs_environment,
    ) = create_custom_environment()

    robot.pose.position = np.array([-3, 0.3])

    # robot.pose.position = np.array([-3, 0.0])

    # A random gamma-check
    gammas = np.zeros(2)
    for ii, obs in enumerate(robot.obstacle_environment):
        gammas[ii] = obs.get_gamma(robot.pose.position, in_global_frame=True)
    # breakpoint()

    x_lim = [-11.0, 11.0]
    y_lim = [-11.0, 11.0]

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    mixed_avoider = MixedEnvironmentAvoider(
        robot=robot,
        weight_max_norm=1e9,
        weight_factor=2,
        weight_power=2.0,
        scaling_laserscan_weight=0.8,
        delta_sampling=2 * math.pi / main_environment.n_samples,
    )

    static_visualization_of_sample_avoidance_mixed(
        robot=robot,
        n_resolution=20,
        dynamical_system=initial_dynamics,
        fast_avoider=mixed_avoider,
        plot_initial_robot=True,
        plot_velocities=True,
        plot_norm_dirs=True,
        sample_environment=main_environment,
        # show_ticks=False,
        show_ticks=True,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        do_quiver=True,
    )


def create_custom_environment(control_radius=0.5):
    dimension = 2

    x_lim = [-11.0, 11.0]
    y_lim = [-11.0, 11.0]

    # User defined values
    x_lim_pos = [-9, 9]
    y_start = -8.5
    y_attractor = 8.5

    pos_start = np.array([get_random_position(x_lim_pos), y_start])
    pos_attractor = np.array([get_random_position(x_lim_pos), y_attractor])

    # Random position for initial position and attractor
    # pos_start = get_random_position(x_lim, y_lim)
    # pos_start[0] = x_start

    # Limit attractor range
    # pos_attractor = get_random_position(x_lim, np.array(y_lim)*0.4)
    # pos_attractor[0] = x_attractor

    robot = QoloRobot(pose=ObjectPose(position=pos_start, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = control_radius

    initial_dynamics = LinearSystem(
        attractor_position=pos_attractor, maximum_velocity=1.5
    )

    # Set obstacle environment
    # x_lim_obs = [x_start + axes_max, x_attractor - axes_max]

    main_environment = ShapelySamplingContainer(n_samples=50)
    obs_environment = ObstacleContainer()

    # Boundary cuboid [could be a door]
    main_environment.create_cuboid(
        position=np.array([0, 0]), axes_length=np.array([19, 19]), is_boundary=True
    )

    # Edge obstacle

    width_edge = 8.0
    edge_shape = np.array([width_edge + 0.2, 2.5])
    center_x = 10 - width_edge / 2.0

    obs_environment.append(
        CuboidXd(
            center_position=np.array([center_x, -3]),
            axes_length=edge_shape,
            margin_absolut=robot.control_radius,
            is_boundary=False,
        )
    )

    obs_environment[-1].set_reference_point(
        np.array([9.55, -3]),
        in_obstacle_frame=False,
    )

    main_environment.create_cuboid(obstacle=obs_environment[-1])

    # Edge obstacle
    obs_environment.append(
        CuboidXd(
            center_position=np.array([(-1) * center_x, 3]),
            axes_length=edge_shape,
            margin_absolut=robot.control_radius,
            is_boundary=False,
        )
    )

    obs_environment[-1].set_reference_point(
        np.array([-9.55, 3]),
        in_obstacle_frame=False,
    )

    main_environment.create_cuboid(obstacle=obs_environment[-1])

    # 2x Random ellipses
    axes_min = 1.5
    axes_max = 4.0

    position, orientation_deg, axes_length = get_random_position_orientation_and_axes(
        x_lim=[-10, 0], y_lim=[-7.5, 2.5], axes_range=[axes_min, axes_max]
    )
    main_environment.create_ellipse(
        position=position,
        orientation_in_degree=orientation_deg,
        axes_length=axes_length,
    )

    position, orientation_deg, axes_length = get_random_position_orientation_and_axes(
        x_lim=[0, 10], y_lim=[-2.5, 7.5], axes_range=[axes_min, axes_max]
    )
    main_environment.create_ellipse(
        position=position,
        orientation_in_degree=orientation_deg,
        axes_length=axes_length,
    )

    robot.obstacle_environment = obs_environment

    return robot, initial_dynamics, main_environment, obs_environment


def animation_comparison(
    robot,
    initial_dynamics,
    main_environment=None,
    obstacle_environment=None,
    do_the_plotting=True,
    it_max=500,
):
    # Only robot moves - everything else is static
    robot = copy.deepcopy(robot)

    weight_max_norm = 1e6
    weight_factor = 1.0
    weight_power = 1.0

    if obstacle_environment is None:
        mode_name = "sample"

        # Sample scenario only
        fast_avoider = SampledAvoider(
            robot=robot,
            weight_max_norm=weight_max_norm,
            weight_factor=2 * np.pi / main_environment.n_samples * 10,
            weight_power=weight_power,
            reference_update_before_modulation=True,
        )

        my_animator = LaserscanAnimator(
            it_max=it_max,
            dt_simulation=0.05,
        )
    elif main_environment is None:
        mode_name = "obstacle"

        fast_avoider = FastObstacleAvoider(
            robot=robot,
            obstacle_environment=robot.obstacle_environment,
            # weight_max_norm=weight_max_norm,
            # weight_factor=weight_factor,
            # weight_power=weight_power,
            reference_update_before_modulation=True,
            evaluate_velocity_weight=True,
        )

        my_animator = FastObstacleAnimator(
            it_max=it_max,
            dt_simulation=0.05,
        )

    else:
        mode_name = "mixed"

        fast_avoider = MixedEnvironmentAvoider(
            robot=robot,
            weight_max_norm=weight_max_norm,
            weight_factor=weight_factor,
            weight_power=weight_power,
            reference_update_before_modulation=True,
            delta_sampling=2 * np.pi / main_environment.n_samples * 10,
        )

        my_animator = MixedObstacleAnimator(
            it_max=it_max,
            dt_simulation=0.05,
        )
    my_animator.setup(
        robot=robot,
        initial_dynamics=initial_dynamics,
        avoider=fast_avoider,
        environment=main_environment,
        # x_lim=x_lim,
        # y_lim=y_lim,
        show_ticks=True,
        show_reference=True,
        show_reference_points=True,
        do_the_plotting=do_the_plotting,
    )

    # self = my_animator
    # self.robot.pose.position = np.array([0, 0])
    # data_points = self.environment.get_surface_points(
    # center_position=self.robot.pose.position,
    # null_direction=self.velocity_command,
    # )
    # self.avoider.update_reference_direction(data_points, in_robot_frame=False)
    # self.initial_velocity = self.initial_dynamics.evaluate(self.robot.pose.position)
    # self.modulated_velocity = self.avoider.avoid(self.initial_velocity)

    # if True:
    # breakpoint()
    # return

    if do_the_plotting:
        my_animator.run(save_animation=False)

    else:
        # plt.close("all")
        my_animator.run_without_plotting()

    print(f"Convergence state of {mode_name}: {my_animator.convergence_state}")
    print()

    return my_animator.convergence_state


def main_comparison(
    do_the_plotting=True,
    n_repetitions=10,
):
    # Do a random seed
    np.random.seed(2)

    dimension = 2

    n_modes = 2

    convergence_states = np.zeros((n_modes, n_repetitions))

    for ii in range(n_repetitions):
        # (
        # robot,
        # initial_dynamics,
        # main_environment,
        # obs_environment,
        # ) = create_new_environment()

        # (
        # robot,
        # initial_dynamics,
        # main_environment,
        # obs_environment,
        # ) = create_fourobstacle_environment()

        (
            robot,
            initial_dynamics,
            main_environment,
            obs_environment,
        ) = create_custom_environment()

        # Obstacle Environment
        # animation_comparison(
        # robot=robot,
        # initial_dynamics=initial_dynamics,
        # obstacle_environment=obs_environment,
        # do_the_plotting=do_the_plotting,
        # )

        # Sample Environment
        convergence_states[0, ii] = animation_comparison(
            robot=robot,
            initial_dynamics=initial_dynamics,
            main_environment=main_environment,
            do_the_plotting=do_the_plotting,
        )

        # Mixed Environment
        convergence_states[1, ii] = animation_comparison(
            robot=robot,
            initial_dynamics=initial_dynamics,
            main_environment=main_environment,
            obstacle_environment=obs_environment,
            do_the_plotting=do_the_plotting,
        )

    print(convergence_states)
    return convergence_states


def example_vectorfield(
    save_figure=False,
    n_resolution=10,
    figisze=(4.5, 4),
    # figisze=(9.0, 8),
):
    np.random.seed(2)

    x_lim = [-11, 11]
    y_lim = [-11, 11]

    (
        robot,
        initial_dynamics,
        main_environment,
        obs_environment,
    ) = create_custom_environment()

    # Plot the vectorfield around the robot
    fig, ax = plt.subplots(1, 1, figsize=figisze)

    mixed_avoider = MixedEnvironmentAvoider(
        robot=robot,
        weight_max_norm=1e9,
        weight_factor=1,
        weight_power=1.0,
        # scaling_laserscan_weight=1.0,
        delta_sampling=2 * np.pi / main_environment.n_samples * 15,
    )

    static_visualization_of_sample_avoidance_mixed(
        robot=robot,
        n_resolution=n_resolution,
        dynamical_system=initial_dynamics,
        fast_avoider=mixed_avoider,
        plot_initial_robot=False,
        sample_environment=main_environment,
        show_ticks=False,
        # show_ticks=True,
        do_quiver=False,
        # do_quiver=True,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
    )

    if save_figure:
        figure_name = "custom_environment_for_comparison_mixed"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

    # if True:
    # return

    fig, ax = plt.subplots(1, 1, figsize=figisze)

    sampled_avoider = SampledAvoider(
        robot=robot,
        weight_max_norm=1e9,
        weight_power=1.0,
        weight_factor=2 * np.pi / main_environment.n_samples * 10,
    )

    static_visualization_of_sample_avoidance(
        robot=robot,
        n_resolution=n_resolution,
        dynamical_system=initial_dynamics,
        fast_avoider=sampled_avoider,
        plot_initial_robot=False,
        main_environment=main_environment,
        show_ticks=False,
        # show_ticks=True,
        # do_quiver=False,
        # do_quiver=True,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
    )

    if save_figure:
        figure_name = "custom_environment_for_comparison_sampled"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def example_integrations(
    save_figure=False,
    n_resolution=10,
    figisze=(4.5, 4),
    n_trajectories=9,
    max_it=3000,
    dt=0.1,
    # figisze=(9.0, 8),
):
    np.random.seed(4)

    x_lim = [-11, 11]
    y_lim = [-11, 11]

    (
        robot,
        initial_dynamics,
        main_environment,
        obs_environment,
    ) = create_custom_environment()

    initial_positions = np.linspace([-8, -8], [8, -8], n_trajectories).T

    fig, ax = plt.subplots(1, 1, figsize=figisze)
    mixed_avoider = MixedEnvironmentAvoider(
        robot=robot,
        weight_max_norm=1e9,
        weight_factor=1,
        weight_power=1.0,
        # scaling_laserscan_weight=1.0,
        delta_sampling=2 * np.pi / (main_environment.n_samples),
    )

    visualization_mixed_environment_with_multiple_integration(
        dynamical_system=initial_dynamics,
        start_positions=initial_positions,
        sample_environment=main_environment,
        robot=robot,
        fast_avoider=mixed_avoider,
        max_it=max_it,
        ax=ax,
        x_lim=x_lim,
        y_lim=y_lim,
    )

    if save_figure:
        figure_name = "custom_environment_integration_for_comparison_sampled"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

    fig, ax = plt.subplots(1, 1, figsize=figisze)

    sampled_avoider = SampledAvoider(
        robot=robot,
        weight_max_norm=1e9,
        weight_power=1.0,
        weight_factor=2 * np.pi / (main_environment.n_samples),
    )

    visualization_mixed_environment_with_multiple_integration(
        dynamical_system=initial_dynamics,
        start_positions=initial_positions,
        sample_environment=main_environment,
        robot=robot,
        fast_avoider=sampled_avoider,
        max_it=max_it,
        ax=ax,
        x_lim=x_lim,
        y_lim=y_lim,
    )

    if save_figure:
        figure_name = "custom_environment_integration_for_comparison_mixed"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def evaluation_convergence(convergence_states):
    sum_states = np.sum(convergence_states > 0, axis=1)
    n_runs = convergence_states.shape[1]

    print(f"Run Evaluation with {n_runs} iterations.")
    print()
    print(f"Convergence")
    print(
        f"Sampled : {round(sum_states[0]/n_runs*100, 1)}% "
        + f"| Mixed : {round(sum_states[1]/n_runs*100, 1)}%"
    )
    print()

    ind_succ = np.logical_and(
        convergence_states[0, :] > 0, convergence_states[1, :] > 0
    )
    mean_time = np.mean(convergence_states[:, ind_succ], axis=0)

    print(f"Mean Time")
    print(f"Sampled : {mean_time[0]} | Mixed : {mean_time[1]}")
    print()
    print()


if (__name__) == "__main__":
    # plt.close("all")
    plt.ion()

    # convergence_states = main_comparison(do_the_plotting=True, n_repetitions=1)
    # convergence_states = main_comparison(do_the_plotting=False, n_repetitions=1)
    # convergence_states = main_comparison(do_the_plotting=False, n_repetitions=100)
    # evaluation_convergence(convergence_states)

    # example_vectorfield(n_resolution=100, save_figure=True)
    # example_integrations(save_figure=False)

    visualize_vectorfield_mixed()

    print("Done")
