""" Script to create plots. """
# Author: Lukas Huber
# Created: 2021-02-22
# Email: lukas.huber@epfl.ch

import copy
from timeit import default_timer as timer
from pathlib import Path

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import shapely

from vartools.states import ObjectPose
from vartools.dynamical_systems import ConstantValue, LinearSystem
from vartools.dynamical_systems import plot_dynamical_system_streamplot

from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.sampling_container import visualize_obstacles
from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer
from fast_obstacle_avoidance.sampling_container import SampledEllipse

from fast_obstacle_avoidance.visualization import LaserscanAnimator
from fast_obstacle_avoidance.visualization import (
    static_visualization_of_sample_avoidance,
)

from fast_obstacle_avoidance.comparison.vfh_avoider import VFH_Avoider_Matlab

# from fast_obstacle_avoidance.comparison.vfh_avoider import VectorFieldHistogramAvoider


def execute_avoidance_with_single_obstacle(save_figure=False, create_animation=False):
    """Visualizes
    (1) the repulsion from a specific point
    (2) the resulting vector-field."""

    start_point = np.array([-2.5, 3])
    x_lim = [-4, 4.5]
    y_lim = [-1.0, 5.6]

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([3.5, 1.3]), maximum_velocity=1.0
    )

    main_environment = ShapelySamplingContainer(n_samples=100)
    main_environment.add_obstacle(
        SampledEllipse.from_obstacle(
            position=np.array([0.5, 0.5]),
            orientation_in_degree=90,
            axes_length=np.array([4.0, 3.0]),
        )
    )

    robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.6

    fast_avoider = VFH_Avoider_Matlab(
        robot=robot,
        # matlab_engine=matlab_eng,
    )

    # fast_avoider = VectorFieldHistogramAvoider(
    #     # attractor_position=
    #     robot=robot,
    # )

    # Do the animation, only:
    if create_animation:
        simu_environment = copy.deepcopy(main_environment)
        simu_environment.n_samples = 40

        simu_bot = copy.deepcopy(robot)
        simu_bot.pose.position = np.array([-2.5, 1])

        my_animator = LaserscanAnimator(
            it_max=400,
            dt_simulation=0.05,
            # dt_pause=0.1,
            animation_name="single_obstacle_avoidance_sampled",
        )

        fast_avoider.robot = simu_bot
        fast_avoider.weight_factor = 2 * np.pi / main_environment.n_samples * 1
        fast_avoider.weight_power = 0.5
        fast_avoider.weight_max_norm = 1e7

        my_animator.setup(
            robot=simu_bot,
            initial_dynamics=initial_dynamics,
            avoider=fast_avoider,
            environment=simu_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            plot_lidarlines=True,
            show_reference=False,
            show_lidarweight=False,
            show_ticks=True,
        )

        my_animator.run(save_animation=save_figure)

        print(f"Done the animation. Saved={save_figure}.")
        return


def vectorfield_with_many_obstacles(save_figure=False, create_animation=True):
    start_point = np.array([-1, 1])
    x_lim = [-8, 4]
    y_lim = [-0.9, 5.6]

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([3.5, 1.3]), maximum_velocity=1.0
    )

    main_environment = ShapelySamplingContainer(n_samples=50)

    # Ellipse
    main_environment.create_ellipse(
        position=np.array([-3, 3.5]),
        orientation_in_degree=50,
        axes_length=np.array([2.0, 0.8]),
    )

    # Ellipse
    main_environment.create_ellipse(
        position=np.array([0.2, 1.1]),
        orientation_in_degree=-20,
        axes_length=np.array([2.0, 1.6]),
    )

    # Box
    main_environment.create_cuboid(geometry=shapely.geometry.box(-6, -1, -5, 1.5))

    # Second Box
    main_environment.create_cuboid(geometry=shapely.geometry.box(0, 4, 1, 8))

    robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.6

    fast_avoider = SampledAvoider(
        robot=robot,
        weight_max_norm=1e8,
        weight_factor=0.1,
        weight_power=3.0,
        evaluate_velocity_weight=True,
    )

    if create_animation:
        simu_environment = copy.deepcopy(main_environment)
        simu_environment.n_samples = 100

        simu_bot = copy.deepcopy(robot)
        simu_bot.pose.position = np.array([-7.5, 0.8])

        my_animator = LaserscanAnimator(
            it_max=400,
            dt_simulation=0.05,
            animation_name="multi_obstacle_avoidance_sampled",
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
            show_lidarweight=True,
            colobar_pos=[0.74, 0.75, 0.14, 0.02],
        )

        my_animator.run(save_animation=save_figure)
        return


def multiple_random_circles():
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
        main_environment.create_sphere(position=position, radius=human_radius)

    fast_avoider = SampledAvoider(
        robot=robot,
        weight_max_norm=1e10,
        weight_factor=1.0,
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
    start_global_matlab_engine = False
    if start_global_matlab_engine and not "matlab_eng" in locals():
        import matlab
        import matlab.engine

        matlab_eng = matlab.engine.start_matlab()
        matlab_eng.addpath("src/fast_obstacle_avoidance/comparison/matlab")
        # str(Path("src") / "fast_obstacle_avoidance" / "comparison" / "matlab")

    plt.ion()
    plt.close("all")

    execute_avoidance_with_single_obstacle(save_figure=False, create_animation=True)

    # test_multi_obstacles()
    # vectorfield_with_many_obstacles(create_animation=True, save_figure=False)
    # vectorfield_with_many_obstacles(create_animation=True, save_figure=False)
    # vectorfield_with_many_obstacles(save_figure=False)

    # test_various_surface_points()
    # multiple_random_circles()

    print("Done.")
