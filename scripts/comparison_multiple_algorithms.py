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
from dynamic_obstacle_avoidance.obstacles import CircularObstacle

# from dynamic_obstacle_avoidance.containers import ObstacleContainer
# from dynamic_obstacle_avoidance.containers import GradientContainer
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


def create_new_environment(control_radius=0.5):
    dimension = 2

    # User defined values
    x_start = -7
    x_attractor = 7
    y_vals = [-4, 4]

    axes_min = 0.5
    axes_max = 2

    # Set robot and start position
    pos_start = np.zeros(dimension)
    pos_start[0] = x_start
    pos_start[1] = np.random.rand(1) * (y_vals[1] - y_vals[0]) + y_vals[0]

    robot = QoloRobot(pose=ObjectPose(position=pos_start, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = control_radius

    # Set dynamical system
    pos_attractor = np.zeros(dimension)
    pos_attractor[0] = x_attractor
    pos_attractor[1] = np.random.rand(1) * (y_vals[1] - y_vals[0]) + y_vals[0]

    initial_dynamics = LinearSystem(
        attractor_position=pos_attractor, maximum_velocity=1.0
    )

    # Set obstacle environment
    x_lim_obs = [x_start + axes_max, x_attractor - axes_max]

    main_environment = ShapelySamplingContainer(n_samples=50)

    # Get the index of the real obstacles
    n_tot = 4
    n_real = 4
    human_radius = 0.7
    real_obs_index = np.zeros(n_tot).astype(bool)
    real_obs_index[np.random.choice(n_tot, n_real)] = True

    n_humans = 6

    obs_environment = SphereContainer()

    for ii in range(n_humans):
        # Random ellipse
        position = get_random_position(x_lim=x_lim_obs, y_lim=y_vals)
        main_environment.create_sphere(position, radius=human_radius)

        obs_environment.append(
            CircularObstacle(
                center_position=position,
                radius=human_radius,
                margin_absolut=robot.control_radius,
            )
        )

    # Update reference points
    obs_environment.update_reference_points()

    # Random ellipse
    position, orientation_deg, axes_length = get_random_position_orientation_and_axes(
        x_lim=x_lim_obs, y_lim=y_vals, axes_range=[axes_min, axes_max]
    )
    main_environment.create_ellipse(position, axes_length, orientation_deg)

    # Random cuboid
    position, orientation_deg, axes_length = get_random_position_orientation_and_axes(
        x_lim=x_lim_obs, y_lim=y_vals, axes_range=[axes_min, axes_max]
    )
    main_environment.create_cuboid(position, axes_length, orientation_deg)

    return robot, initial_dynamics, main_environment, obs_environment


def animation_comparison(
    robot, initial_dynamics, main_environment=None, obstacle_environment=None
):
    # Only robot moves - everything else is static
    robot = copy.deepcopy(robot)

    weight_max_norm = 1e8
    weight_factor = 4
    weight_power = 2.0
    if obstacle_environment is None:
        # Sample scenario only
        fast_avoider = SampledAvoider(
            robot=robot,
            weight_max_norm=weight_max_norm,
            weight_factor=weight_factor,
            weight_power=weight_power,
        )

        my_animator = LaserscanAnimator(
            it_max=400,
            dt_simulation=0.05,
        )
    elif main_environment is None:
        print("Obstacle only.")
        robot.obstacle_environment = obstacle_environment

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
            it_max=400,
            dt_simulation=0.05,
        )

    else:
        print("Doing the mixed.")
        robot.obstacle_environment = obstacle_environment

        fast_avoider = MixedEnvironmentAvoider(
            robot=robot,
            weight_max_norm=weight_max_norm,
            weight_factor=weight_factor,
            weight_power=weight_power,
        )

        my_animator = MixedObstacleAnimator(
            it_max=400,
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
    )

    my_animator.run(save_animation=False)

    if my_animator.convergence_state:
        return (-1) * my_animator.convergence_state
    else:
        return my_animator.ii


def main_comparison():
    # Do a random seed
    np.random.seed(5)

    dimension = 2

    n_modes = 2
    n_repetitions = 1

    convergence_states = np.zeros((n_modes, n_repetitions))

    for ii in range(n_repetitions):
        (
            robot,
            initial_dynamics,
            main_environment,
            obs_environment,
        ) = create_new_environment()

        # convergence_states[1, ii] = animation_comparison(
        # robot=robot,
        # initial_dynamics=initial_dynamics,
        # obstacle_environment=obs_environment,
        # )

        # For now only fix the obstacle
        # if True:
        # continue
        convergence_states[0, ii] = animation_comparison(
            robot=robot,
            initial_dynamics=initial_dynamics,
            main_environment=main_environment,
        )

        convergence_states[1, ii] = animation_comparison(
            robot=robot,
            initial_dynamics=initial_dynamics,
            main_environment=main_environment,
            obstacle_environment=obs_environment,
        )

    print(convergence_states)
    return convergence_states


if (__name__) == "__main__":

    plt.close("all")
    plt.ion()

    convergence_states = main_comparison()
