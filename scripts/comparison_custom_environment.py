""" Script to create plots. """
# Author: Lukas Huber
# Created: 2021-03-02
# Email: lukas.huber@epfl.ch

import copy
from timeit import default_timer as timer
import math
from enum import Enum, auto

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import shapely

# from shapely import affinity

from vartools.states import ObjectPose
from vartools.dynamical_systems import LinearSystem

# from dynamic_obstacle_avoidance.obstacles import Ellipse, Cuboid
from dynamic_obstacle_avoidance.obstacles.ellipse_xd import EllipseWithAxes as Ellipse
from dynamic_obstacle_avoidance.obstacles.cuboid_xd import CuboidXd as Cuboid

from dynamic_obstacle_avoidance.obstacles import CircularObstacle

# from dynamic_obstacle_avoidance.containers import SphereContainer
from dynamic_obstacle_avoidance.containers import ObstacleContainer

# from dynamic_obstacle_avoidance.avoidance import ModulationAvoider
from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer

from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider
from fast_obstacle_avoidance.obstacle_avoider import MixedEnvironmentAvoider
from fast_obstacle_avoidance.obstacle_avoider import ModulationAvoider

from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.visualization import MixedObstacleAnimator
from fast_obstacle_avoidance.visualization import FastObstacleAnimator
from fast_obstacle_avoidance.visualization import LaserscanAnimator

from fast_obstacle_avoidance.visualization.integration_plot import (
    visualization_mixed_environment_with_multiple_integration,
)

from fast_obstacle_avoidance.visualization import (
    static_visualization_of_sample_avoidance_mixed,
    static_visualization_of_sample_avoidance,
    # static_visualization_of_sample_avoidance_obstacle,
)

# Comparison algorithm inspired on matlab
from fast_obstacle_avoidance.comparison.vfh_avoider import VFH_Avoider


class AlgorithmType(Enum):
    SAMPLED = 0
    MIXED = 1
    VFH = 2
    OBSTACLE = auto()
    MODULATED = auto()


def get_random_position_orientation_and_axes(x_lim, y_lim, axes_range):
    dimension = 2

    position = np.random.rand(dimension)
    position[0] = position[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    position[1] = position[1] * (y_lim[1] - y_lim[0]) + y_lim[0]

    # orientation_deg = np.random.rand(1) * 360
    orientation = np.random.rand(1)[0] * math.pi

    axes_length = np.random.rand(2) * (axes_range[1] - axes_range[0]) + axes_range[0]

    return position, orientation, axes_length


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

    np.random.seed(2)
    (
        robot,
        initial_dynamics,
        main_environment,
        obs_environment,
        _,
    ) = create_custom_environment()

    # robot.pose.position = np.array([-3, 0.3])
    robot.pose.position = np.array([5, -5.5])

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
        weight_factor=3.0,
        weight_power=1.0,
        scaling_laserscan_weight=1.5,
        delta_sampling=2 * math.pi / (main_environment.n_samples),
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


def visualize_vectorfield_sampled():
    """Visualize vectorfield of mixed environment."""
    np.random.seed(2)

    (
        robot,
        initial_dynamics,
        main_environment,
        obs_environment,
        _,
    ) = create_custom_environment()

    # robot.pose.position = np.array([-3, 0.3])
    robot.pose.position = np.array([5, -5.5])

    # A random gamma-check
    gammas = np.zeros(2)
    for ii, obs in enumerate(robot.obstacle_environment):
        gammas[ii] = obs.get_gamma(robot.pose.position, in_global_frame=True)
    # breakpoint()

    x_lim = [-11.0, 11.0]
    y_lim = [-11.0, 11.0]

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    fast_avoider = SampledAvoider(
        robot=robot,
        weight_max_norm=1e9,
        weight_factor=2,
        weight_power=4.0,
        # scaling_laserscan_weight=0.1,
    )

    static_visualization_of_sample_avoidance(
        robot=robot,
        n_resolution=20,
        dynamical_system=initial_dynamics,
        fast_avoider=fast_avoider,
        plot_initial_robot=True,
        plot_velocities=True,
        # plot_norm_dirs=True,
        main_environment=main_environment,
        # show_ticks=False,
        show_ticks=True,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        plot_quiver=True,
    )


def visualize_vectorfield_full():
    """Visualize vectorfield of mixed environment."""
    np.random.seed(2)

    (
        robot,
        initial_dynamics,
        main_environment,
        partial_environment,
        full_environment,
    ) = create_custom_environment()

    # robot.pose.position = np.array([-3, 0.3])
    robot.pose.position = np.array([5, -5.5])

    # A random gamma-check
    gammas = np.zeros(2)
    for ii, obs in enumerate(robot.obstacle_environment):
        gammas[ii] = obs.get_gamma(robot.pose.position, in_global_frame=True)
    # breakpoint()

    x_lim = [-11.0, 11.0]
    y_lim = [-11.0, 11.0]

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    fast_avoider = FastObstacleAvoider(
        obstacle_environment=full_environment,
        robot=robot,
        weight_max_norm=1e5,
        weight_factor=5,
        weight_power=2.5,
        # scaling_laserscan_weight=0.1,
    )

    fast_avoider = ModulationAvoider()

    breakpoint()

    static_visualization_of_sample_avoidance(
        robot=robot,
        n_resolution=20,
        dynamical_system=initial_dynamics,
        fast_avoider=fast_avoider,
        plot_initial_robot=True,
        plot_velocities=True,
        # plot_norm_dirs=True,
        main_environment=main_environment,
        # show_ticks=False,
        show_ticks=True,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        plot_quiver=True,
    )


def create_custom_environment(
    control_radius=0.5,
):
    dimension = 2

    x_lim = [-11.0, 11.0]
    y_lim = [-11.0, 11.0]

    # User defined values
    x_lim_pos = [-9, 9]
    y_start = -8.5
    y_attractor = 8.0

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
    full_environment = ObstacleContainer()

    # Boundary cuboid [could be a door]
    full_environment.append(
        Cuboid(
            center_position=np.array([0, 0]),
            axes_length=np.array([19, 19]),
            margin_absolut=robot.control_radius,
            is_boundary=True,
        )
    )
    main_environment.create_cuboid(
        position=np.array([0, 0]), axes_length=np.array([19, 19]), is_boundary=True
    )
    # main_environment.create_cuboid(full_environment.append[-1])

    # Edge obstacle #1
    width_edge = 8.0
    edge_shape = np.array([width_edge + 0.2, 2.5])
    center_x = 10 - width_edge / 2.0

    obs_environment.append(
        Cuboid(
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

    # Copy the elements
    full_environment.append(obs_environment[-1])
    main_environment.create_cuboid(obstacle=obs_environment[-1])

    # Edge obstacle #2
    obs_environment.append(
        Cuboid(
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

    # Copy the elements
    full_environment.append(obs_environment[-1])
    main_environment.create_cuboid(obstacle=obs_environment[-1])

    # 2x Random ellipses [# 1]
    axes_min = 1.5
    axes_max = 4.0

    position, orientation, axes_length = get_random_position_orientation_and_axes(
        x_lim=[-10, 0], y_lim=[-7.5, 2.5], axes_range=[axes_min, axes_max]
    )

    full_environment.append(
        Ellipse(
            center_position=position,
            orientation=orientation,
            axes_length=axes_length,
        )
    )
    main_environment.create_ellipse(
        obstacle=full_environment[-1]
        # position=position,
        # orientation=orientation,
        # axes_length=axes_length,
    )

    # 2x Random ellipses [# 1]
    position, orientation_deg, axes_length = get_random_position_orientation_and_axes(
        x_lim=[0, 10], y_lim=[-2.5, 7.5], axes_range=[axes_min, axes_max]
    )

    full_environment.append(
        Ellipse(
            center_position=position,
            orientation=orientation,
            # orientation_in_degree=orientation_deg,
            axes_length=axes_length,
        )
    )

    main_environment.create_ellipse(
        obstacle=full_environment[-1]
        # position=position,
        # orientation=orientation,
        # orientation_in_degree=orientation_deg,
        # axes_length=axes_length,
    )

    robot.obstacle_environment = obs_environment

    return robot, initial_dynamics, main_environment, obs_environment, full_environment


def animation_comparison(
    robot,
    initial_dynamics,
    mode_type: AlgorithmType,
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

    if mode_type == AlgorithmType.SAMPLED:
        # mode_name = "sample"

        # Sample scenario only
        fast_avoider = SampledAvoider(
            robot=robot,
            weight_max_norm=1e9,
            weight_factor=2,
            weight_power=4.0,
            #     robot=robot,
            #     weight_max_norm=weight_max_norm,
            #     weight_factor=2 * np.pi / main_environment.n_samples * 10,
            #     weight_power=weight_power,
            #     reference_update_before_modulation=True,
        )

        my_animator = LaserscanAnimator(
            it_max=it_max,
            dt_simulation=0.05,
        )
    elif mode_type == AlgorithmType.OBSTACLE:
        # mode_name = "obstacle"

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

    elif mode_type == AlgorithmType.VFH:
        fast_avoider = VFH_Avoider(robot=robot)

        my_animator = LaserscanAnimator(
            it_max=it_max,
            dt_simulation=0.05,
        )

    elif mode_type == AlgorithmType.MIXED:
        fast_avoider = MixedEnvironmentAvoider(
            # robot=robot,
            # weight_max_norm=weight_max_norm,
            # weight_factor=weight_factor,
            # weight_power=weight_power,
            # reference_update_before_modulation=True,
            # delta_sampling=2 * np.pi / main_environment.n_samples * 10,
            robot=robot,
            weight_max_norm=1e9,
            weight_factor=3.0,
            weight_power=1.0,
            scaling_laserscan_weight=1.5,
            delta_sampling=2 * math.pi / (main_environment.n_samples),
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
        convergence_distance=5e-1,
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

    if do_the_plotting:
        my_animator.run(save_animation=False)

    else:
        # plt.close("all")
        my_animator.run_without_plotting()

    print(f"Convergence state of {mode_type}: {my_animator.convergence_state}")
    return my_animator


class Datahandler:
    def __init__(self, n_modes, n_runs):
        self.n_runs = n_runs

        # Create buckets
        self.convergence_counter = np.zeros(n_modes)
        self.computation_times = np.zeros((n_modes, 0))
        self.distances = np.zeros((n_modes, 0))
        self.velocities_mean = np.zeros((n_modes, 0))
        self.velocities_deviation = np.zeros((n_modes, 0))
        self.animator_names = [None for _ in range(n_modes)]


def main_comparison(
    do_the_plotting=True,
    n_repetitions=10,
):
    # Do a random seed
    np.random.seed(10)

    dimension = 2

    # n_modes = 2
    n_modes = 3

    # convergence_states = np.zeros((n_modes, n_repetitions))

    dh = Datahandler(n_modes=n_modes, n_runs=n_repetitions)

    animators = [None for _ in range(n_modes)]

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
            _,
        ) = create_custom_environment()
        #

        # Sample Environment
        ii = 0
        dh.animator_names[ii] = "Sampled"
        animators[ii] = animation_comparison(
            mode_type=AlgorithmType.SAMPLED,
            robot=robot,
            initial_dynamics=initial_dynamics,
            main_environment=main_environment,
            do_the_plotting=do_the_plotting,
        )

        # Mixed Environment
        ii += 1
        dh.animator_names[ii] = "Disparate"
        animators[ii] = animation_comparison(
            mode_type=AlgorithmType.MIXED,
            robot=robot,
            initial_dynamics=initial_dynamics,
            main_environment=main_environment,
            obstacle_environment=obs_environment,
            do_the_plotting=do_the_plotting,
        )

        # # VFH Environment
        ii += 1
        dh.animator_names[ii] = "VFH"
        animators[ii] = animation_comparison(
            mode_type=AlgorithmType.VFH,
            robot=robot,
            initial_dynamics=initial_dynamics,
            main_environment=main_environment,
            obstacle_environment=obs_environment,
            do_the_plotting=do_the_plotting,
        )

        # TODO: add additional velocities [Modulated / Baseline]

        conv_states = [(ani.convergence_state > 0) for ani in animators]
        dh.convergence_counter = dh.convergence_counter + conv_states

        # breakpoint()
        if sum(conv_states) == n_modes:
            dh.distances = np.append(
                dh.distances,
                np.array([ani.get_total_distance() for ani in animators]).reshape(
                    -1, 1
                ),
                axis=1,
            )

            dh.computation_times = np.append(
                dh.computation_times,
                np.array(
                    [ani.get_mean_coputation_time_ms() for ani in animators]
                ).reshape(-1, 1),
                axis=1,
            )

        # TODO: add additional velocities []

    # print(convergence_states)
    return dh


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
        _,
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
        fig.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

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
        fig.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


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
        _,
    ) = create_custom_environment()

    initial_positions = np.linspace([-8, -8], [8, -8], n_trajectories).T

    # Do the VFH
    fig, ax = plt.subplots(1, 1, figsize=figisze)
    vfh_avoider = VFH_Avoider(
        robot=robot,
        # use_matlab=True,
        # matlab_engine=matlab_eng,
    )
    visualization_mixed_environment_with_multiple_integration(
        dynamical_system=initial_dynamics,
        start_positions=initial_positions,
        sample_environment=main_environment,
        robot=robot,
        fast_avoider=vfh_avoider,
        max_it=max_it,
        ax=ax,
        x_lim=x_lim,
        y_lim=y_lim,
    )

    if save_figure:
        figure_name = "custom_environment_integration_for_comparison_vfh"
        fig.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

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
        figure_name = "custom_environment_integration_for_comparison_mixed"
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
        figure_name = "custom_environment_integration_for_comparison_sampled"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def evaluation_convergence(dh: Datahandler):
    # sum_states = np.sum(convergence_states > 0, axis=1)
    # n_runs = convergence_state

    newline = " \\\\"
    print(f"Restuls of run Evaluation with {dh.n_runs} iterations.")
    print("")
    print(" & " + " & ".join(dh.animator_names) + newline)

    if dh.n_runs:
        pp_conv = np.round(dh.convergence_counter / n_runs * 100, 1)
    else:
        pp_conv = np.zeros_like(dh.convergence_counter)

    print(
        f"Convergence Ratio & "
        + f" & ".join([f"{pp_conv[ii]}\\%" for ii in range(pp_conv.shape[0])])
        + newline
    )

    dd_mean = np.round(np.mean(dh.distances, axis=1), 2)
    dd_std = np.round(np.std(dh.distances, axis=1), 2)
    print(
        f"Distance [m] & "
        + " & ".join(
            [f"{dd_mean[ii]} \pm {dd_std[ii]}" for ii in range(dd_mean.shape[0])]
        )
        + newline
    )

    ct_mean = np.round(np.mean(10 * dh.computation_times, axis=1), 1)
    ct_std = np.round(np.std(10 * dh.computation_times, axis=1), 1)
    print(
        f"Comp. Time [1e-4 s] & "
        + " & ".join(
            [f"{ct_mean[ii]} \pm {ct_std[ii]}" for ii in range(ct_mean.shape[0])]
        )
        + newline
    )

    # breakpoint()


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # n_runs = 100
    n_runs = 5

    # convergence_states = main_comparison(do_the_plotting=True, n_repetitions=1)
    # c_count, dist, t_comp = main_comparison(do_the_plotting=False, n_repetitions=n_runs)

    # datahandler = main_comparison(do_the_plotting=False, n_repetitions=n_runs)
    # evaluation_convergence(datahandler)

    # example_vectorfield(n_resolution=100, save_figure=True)
    # example_integrations(save_figure=False)

    # visualize_vectorfield_mixed()
    # visualize_vectorfield_sampled()
    visualize_vectorfield_full()

    print("")
    print("Done")
