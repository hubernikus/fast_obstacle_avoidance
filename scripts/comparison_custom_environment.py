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

from dynamic_obstacle_avoidance.obstacles import Ellipse, Cuboid
from dynamic_obstacle_avoidance.obstacles import CircularObstacle

from dynamic_obstacle_avoidance.containers import SphereContainer

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer

from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider
from fast_obstacle_avoidance.obstacle_avoider import MixedEnvironmentAvoider

from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.visualization import MixedObstacleAnimator
from fast_obstacle_avoidance.visualization import FastObstacleAnimator
from fast_obstacle_avoidance.visualization import LaserscanAnimator


def get_random_position_orientation_and_axes(x_lim, y_lim, axes_range):
    dimension = 2

    position = np.random.rand(dimension)
    position[0] = position[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    position[1] = position[1] * (y_lim[1] - y_lim[0]) + y_lim[0]

    orientation_deg = np.random.rand(1) * 360

    axes_length = np.random.rand(2) * (axes_range[1] - axes_range[0]) + axes_range[0]

    return position, orientation_deg, axes_length


def get_random_position(x_lim, y_lim):
    dimension = 2

    position = np.random.rand(dimension)
    position[0] = position[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    position[1] = position[1] * (y_lim[1] - y_lim[0]) + y_lim[0]

    return position


def create_custom_environment(control_radius=0.5):
    dimension = 2

    x_lim = [-10, 10]
    y_lim = [-10, 10]

    # User defined values
    pos_start = np.array([-6, -4.5])
    pos_attractor = np.array([6, 4.5])

    x_start = -9.0
    x_attractor = 9.0

    # Random position for initial position and attractor
    # pos_start = get_random_position(x_lim, y_lim)
    # pos_start[0] = x_start

    # Limit attractor range
    # pos_attractor = get_random_position(x_lim, np.array(y_lim)*0.4)
    # pos_attractor[0] = x_attractor

    axes_min = 0.4
    axes_max = 2.0

    robot = QoloRobot(pose=ObjectPose(position=pos_start, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = control_radius

    initial_dynamics = LinearSystem(
        attractor_position=pos_attractor, maximum_velocity=2.0
    )

    # Set obstacle environment
    x_lim_obs = [x_start + axes_max, x_attractor - axes_max]

    main_environment = ShapelySamplingContainer(n_samples=50)
    obs_environment = SphereContainer()

    rand_it_max = 200

    # Get the index of the real obstacles

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
            weight_factor=2 * np.pi / main_environment.n_samples * 3.5,
            weight_power=weight_power,
            reference_update_before_modulation=False,
        )

        my_animator = LaserscanAnimator(
            it_max=it_max,
            dt_simulation=0.05,
        )
    elif main_environment is None:
        mode_name = "obstacle"
        robot.obstacle_environment = obstacle_environment

        fast_avoider = FastObstacleAvoider(
            robot=robot,
            obstacle_environment=robot.obstacle_environment,
            # weight_max_norm=weight_max_norm,
            # weight_factor=weight_factor,
            # weight_power=weight_power,
            reference_update_before_modulation=False,
            evaluate_velocity_weight=True,
        )

        my_animator = FastObstacleAnimator(
            it_max=it_max,
            dt_simulation=0.05,
        )

    else:
        mode_name = "mixed"
        robot.obstacle_environment = obstacle_environment

        fast_avoider = MixedEnvironmentAvoider(
            robot=robot,
            weight_max_norm=weight_max_norm,
            weight_factor=weight_factor,
            weight_power=weight_power,
            reference_update_before_modulation=False,
            delta_sampling=2 * np.pi / main_environment.n_samples,
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
        show_reference=True,
        show_reference_points=True,
        do_the_plotting=do_the_plotting,
    )

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
    n_repetitions=1,
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


if (__name__) == "__main__":

    plt.close("all")
    plt.ion()

    convergence_states = main_comparison()
