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
from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes
from dynamic_obstacle_avoidance.obstacles import CuboidXd


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


def create_custom_environment(control_radius=0.5):
    dimension = 2

    x_lim = [-10, 10]
    y_lim = [-10, 10]

    # User defined values
    # pos_start = np.array([-6, -4.5])
    # pos_attractor = np.array([6, 4.5])

    x_start = -9.5
    x_attractor = 9.5

    # Random position for initial position and attractor
    pos_start = get_random_position(x_lim, y_lim)
    pos_start[0] = x_start

    # Limit attractor range
    pos_attractor = get_random_position(x_lim, np.array(y_lim) * 0.4)
    pos_attractor[0] = x_attractor

    axes_min = 0.8
    axes_max = 4.0

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
    n_tot = 4
    n_real = 2
    real_obs_index = np.zeros(n_tot).astype(bool)
    real_obs_index[np.random.choice(n_tot, n_real)] = True

    axes_length_star = [1.6, 5.0]

    pos_list = []
    rad_list = []

    # Star 1
    position_ref = np.array([0.0, 0])

    obs_environment.append(
        EllipseWithAxes(
            center_position=position_ref + np.array([0, axes_length_star[1] / 2]),
            orientation=30 * np.pi / 180,
            axes_length=axes_length_star,
            margin_absolut=robot.control_radius,
        )
    )

    # Copy to main-environment
    main_environment.create_ellipse(obstacle=obs_environment[-1])

    # Star 2
    obs_environment.append(
        EllipseWithAxes(
            center_position=position_ref - np.array([0, axes_length_star[1] / 2]),
            orientation=-30 * np.pi / 180,
            axes_length=axes_length_star,
            margin_absolut=robot.control_radius,
        )
    )
    # Set common reference point
    exact_ref_point = position_ref + np.array([1, 0])

    obs_environment[-1].set_reference_point(exact_ref_point, in_global_frame=True)
    obs_environment[-2].set_reference_point(exact_ref_point, in_global_frame=True)

    main_environment.create_ellipse(obstacle=obs_environment[-1])

    position, orientation_deg, axes_length = get_random_position_orientation_and_axes(
        x_lim=x_lim, y_lim=y_lim, axes_range=[axes_min, axes_max]
    )

    position[0] = -6.5
    main_environment.create_ellipse(position, axes_length, orientation_deg)

    position, orientation_deg, axes_length = get_random_position_orientation_and_axes(
        x_lim=x_lim, y_lim=y_lim, axes_range=[axes_min, axes_max]
    )

    position[0] = 0
    main_environment.create_cuboid(position, axes_length, orientation_deg)

    return robot, initial_dynamics, main_environment, obs_environment


def create_fourobstacle_environment(control_radius=0.5):
    dimension = 2
    y_vals = [-5, 5]

    # User defined values
    pos_start = np.array([-6, -4.5])
    pos_attractor = np.array([6, 4.5])

    x_start = pos_start[0]
    x_attractor = pos_attractor[0]

    axes_min = 0.4
    axes_max = 2.0

    robot = QoloRobot(pose=ObjectPose(position=pos_start, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = control_radius

    initial_dynamics = LinearSystem(
        attractor_position=pos_attractor, maximum_velocity=1.0
    )

    # Set obstacle environment
    x_lim_obs = [x_start + axes_max, x_attractor - axes_max]

    main_environment = ShapelySamplingContainer(n_samples=50)
    obs_environment = SphereContainer()

    rand_it_max = 200

    # Get the index of the real obstacles
    n_tot = 4
    n_real = 2
    real_obs_index = np.zeros(n_tot).astype(bool)
    real_obs_index[np.random.choice(n_tot, n_real)] = True

    axes_length_star = [2.8, 0.2]

    pos_list = []
    rad_list = []

    # Star 1
    position, orientation_deg, axes_length = get_random_position_orientation_and_axes(
        x_lim=x_lim_obs, y_lim=y_vals, axes_range=[axes_min, axes_max]
    )
    main_environment.create_ellipse(position, axes_length_star, orientation_deg)
    obs_environment.append(
        Ellipse(
            center_position=position,
            orientation=orientation_deg * np.pi / 180,
            axes_length=axes_length_star,
            margin_absolut=robot.control_radius,
        )
    )

    orientation_deg += 90

    main_environment.create_ellipse(position, axes_length_star, orientation_deg)
    obs_environment.append(
        Ellipse(
            center_position=position,
            orientation=orientation_deg * np.pi / 180,
            axes_length=axes_length_star,
            margin_absolut=robot.control_radius,
        )
    )

    pos_list.append(position)
    rad_list.append(max(axes_length_star))

    # Star 2
    for ii in range(rand_it_max):
        (
            position,
            orientation_deg,
            axes_length,
        ) = get_random_position_orientation_and_axes(
            x_lim=x_lim_obs, y_lim=y_vals, axes_range=[axes_min, axes_max]
        )

        distances = np.zeros(len(pos_list))
        for pp in range(len(pos_list)):
            distances[pp] = (
                LA.norm(pos_list[pp] - position)
                - rad_list[pp]
                - max(axes_length_star)
                - 2 * robot.control_radius
            )
            # distances[pp] = LA.norm(pos_list[pp]-position) - rad_list[pp] - max(axes_length_star)

        if all(distances > 0):
            break

    pos_list.append(position)
    rad_list.append(max(axes_length_star))

    main_environment.create_ellipse(position, axes_length_star, orientation_deg)
    obs_environment.append(
        Ellipse(
            center_position=position,
            orientation=orientation_deg * np.pi / 180,
            axes_length=axes_length_star,
            margin_absolut=robot.control_radius,
        )
    )

    orientation_deg += 90

    main_environment.create_ellipse(position, axes_length_star, orientation_deg)
    obs_environment.append(
        Ellipse(
            center_position=position,
            orientation=orientation_deg * np.pi / 180,
            axes_length=axes_length_star,
            margin_absolut=robot.control_radius,
        )
    )

    # Ellipse
    for ii in range(rand_it_max):
        (
            position,
            orientation_deg,
            axes_length,
        ) = get_random_position_orientation_and_axes(
            x_lim=x_lim_obs, y_lim=y_vals, axes_range=[axes_min, axes_max]
        )
        distances = np.zeros(len(pos_list))
        for pp in range(len(pos_list)):
            distances[pp] = (
                LA.norm(pos_list[pp] - position)
                - rad_list[pp]
                - max(axes_length)
                - 2 * robot.control_radius
            )

        if all(distances > 0):
            break

    pos_list.append(position)
    rad_list.append(max(axes_length))

    main_environment.create_ellipse(position, axes_length, orientation_deg)

    # Random cuboid
    for ii in range(rand_it_max):
        (
            position,
            orientation_deg,
            axes_length,
        ) = get_random_position_orientation_and_axes(
            x_lim=x_lim_obs, y_lim=y_vals, axes_range=[axes_min, axes_max]
        )
        distances = np.zeros(len(pos_list))
        for pp in range(len(pos_list)):
            distances[pp] = (
                LA.norm(pos_list[pp] - position)
                - rad_list[pp]
                - max(axes_length)
                - 2 * robot.control_radius
            )

        if all(distances > 0):
            break

    pos_list.append(position)
    rad_list.append(max(axes_length))

    # position, orientation_deg, axes_length = get_random_position_orientation_and_axes(
    # x_lim=x_lim_obs, y_lim=y_vals, axes_range=[axes_min, axes_max]
    # )
    main_environment.create_cuboid(position, axes_length, orientation_deg)

    return robot, initial_dynamics, main_environment, obs_environment


def create_multi_human_environment(control_radius=0.5):
    dimension = 2

    y_vals = [-5, 5]

    # User defined values
    pos_start = np.array([-6, -4.5])
    pos_attractor = np.array([6, 4.5])

    x_start = pos_start[0]
    x_attractor = pos_attractor[0]

    axes_min = 0.5
    axes_max = 2

    # Set robot and start position
    # pos_start = np.zeros(dimension)
    # pos_start[0] = x_start
    # pos_start[1] = np.random.rand(1) * (y_vals[1] - y_vals[0]) + y_vals[0]

    robot = QoloRobot(pose=ObjectPose(position=pos_start, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = control_radius

    # Set dynamical system
    # pos_attractor = np.zeros(dimension)
    # pos_attractor[0] = x_attractor
    # pos_attractor[1] = np.random.rand(1) * (y_vals[1] - y_vals[0]) + y_vals[0]

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
            # reference_update_before_modulation=,
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
            # reference_update_before_modulation=False,
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
            # reference_update_before_modulation=False,
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
