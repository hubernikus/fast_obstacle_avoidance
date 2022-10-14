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
from fast_obstacle_avoidance.sampling_container import SampledEllipse, SampledCuboid

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


def execute_avoidance_through_gap(save_figure=False, create_animation=False):
    """Visualizes
    (1) the repulsion from a specific point
    (2) the resulting vector-field."""

    start_point = np.array([-2.0, 0])
    # start_point = np.array([-2.0, 4])
    x_lim = [-4, 4.5]
    y_lim = [-1.0, 5.6]

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([3.5, 1.3]), maximum_velocity=1.0
    )

    main_environment = ShapelySamplingContainer(n_samples=200)
    main_environment.add_obstacle(
        SampledCuboid.from_obstacle(
            position=np.array([0.5, 0.2]),
            orientation_in_degree=0,
            axes_length=np.array([1.0, 3.0]),
        )
    )

    main_environment.add_obstacle(
        SampledCuboid.from_obstacle(
            position=np.array([0.5, 4.6]),
            orientation_in_degree=0,
            axes_length=np.array([1.0, 3.0]),
        )
    )

    robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.6

    # fast_avoider = VFH_Avoider_Matlab(
    #     robot=robot,
    #     # matlab_engine=matlab_eng,
    # )

    fast_avoider = SampledAvoider(
        robot=robot,
        # weight_max_norm=1e6,
        # weight_factor=2 * np.pi / main_environment.n_samples * 2,
        # weight_power=1.5,
    )

    fast_avoider.robot = robot
    fast_avoider.weight_factor = 2.0 * 2 * np.pi / main_environment.n_samples * 1
    fast_avoider.weight_power = 1.5
    fast_avoider.weight_max_norm = 1e8
    # weight_max_norm=1e8,
    #  weight_factor=2 * np.pi / main_environment.n_samples * 2,
    #  weight_power=1.5,

    # fast_avoider = VectorFieldHistogramAvoider(
    #     # attractor_position=
    #     robot=robot,
    # )

    # Do the animation, only:
    if create_animation:
        simu_environment = copy.deepcopy(main_environment)
        # simu_environment.n_samples = 40

        simu_bot = copy.deepcopy(robot)
        simu_bot.pose.position = start_point

        my_animator = LaserscanAnimator(
            it_max=400,
            dt_simulation=0.05,
            # dt_pause=0.1,
            animation_name="single_obstacle_avoidance_sampled",
        )

        my_animator.setup(
            robot=simu_bot,
            initial_dynamics=initial_dynamics,
            avoider=fast_avoider,
            environment=simu_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            plot_lidarlines=True,
            show_reference=True,
            show_lidarweight=False,
            show_ticks=True,
        )

        my_animator.run(save_animation=save_figure)


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

    # execute_avoidance_with_single_obstacle(save_figure=False, create_animation=True)
    execute_avoidance_through_gap(save_figure=False, create_animation=True)

    print("Done.")
