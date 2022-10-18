""" Script to create plots. """
# Author: Lukas Huber
# Created: 2021-02-22
# Email: lukas.huber@epfl.ch

import copy
from timeit import default_timer as timer

import math

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import shapely

from vartools.states import ObjectPose
from vartools.dynamical_systems import ConstantValue, LinearSystem
from vartools.dynamical_systems import plot_dynamical_system_streamplot

from dynamic_obstacle_avoidance.containers import GradientContainer

from dynamic_obstacle_avoidance.obstacles import CircularObstacle
from dynamic_obstacle_avoidance.visualization import plot_obstacles

# from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider
from fast_obstacle_avoidance.obstacle_avoider import MixedEnvironmentAvoider
from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer
from fast_obstacle_avoidance.sampling_container import SampledEllipse
from fast_obstacle_avoidance.sampling_container import SampledCuboid
from fast_obstacle_avoidance.sampling_container import visualize_obstacles


from fast_obstacle_avoidance.visualization import FastObstacleAnimator
from fast_obstacle_avoidance.visualization import MixedObstacleAnimator
from fast_obstacle_avoidance.visualization import (
    static_visualization_of_sample_avoidance_mixed,
)


def vectorfield_with_scenario_mixed(save_figure=False):
    # start_point = np.array([9, 6])
    start_point = np.array([13, 4])
    # start_point = np.array([3.5, 2.5])
    x_lim = [0, 18]
    y_lim = [0, 8]
    # x_lim = [5, 7]
    # y_lim = [5.5, 7]

    control_radius = 0.6

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([17, 3.0]), maximum_velocity=1.0
    )

    analytic_environment = GradientContainer()
    analytic_environment.append(
        CircularObstacle(
            center_position=np.array([14, 7]),
            orientation=-20 * np.pi / 180,
            radius=0.5,
            margin_absolut=control_radius,
            linear_velocity=np.array([-0.3, -0.5]),
        )
    )

    analytic_environment.append(
        CircularObstacle(
            center_position=np.array([6, 1]),
            orientation=-20 * np.pi / 180,
            radius=0.5,
            margin_absolut=control_radius,
            # linear_velocity=np.array([-0.5, 0.2]),
        )
    )

    sampled_environment = ShapelySamplingContainer(n_samples=50)
    # ellipse = shapely.affinity.scale(shapely.geometry.Point(14, 1).buffer(1), 2, 1)
    # ellipse = shapely.affinity.rotate(ellipse, 50)
    sampled_environment.add_obstacle(
        SampledEllipse.from_obstacle(
            position=np.array([14, 1]),
            orientation_in_degree=50,
            axes_length=np.array([2, 1]),
        )
    )

    # sampled_environment.add_obstacle(shapely.geometry.box(5, 4, 7, 8))
    sampled_environment.add_obstacle(
        SampledCuboid.from_obstacle(
            position=np.array([6, 6]),
            orientation_in_degree=00,
            axes_length=np.array([2, 4]),
        )
    )

    robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = control_radius

    robot.obstacle_environment = analytic_environment

    # Initialize Avoider
    mixed_avoider = MixedEnvironmentAvoider(
        robot=robot,
        weight_max_norm=1e9,
        weight_factor=2,
        weight_power=4.0,
        scaling_laserscan_weight=1.1,
        delta_sampling=2 * math.pi / sampled_environment.n_samples,
    )

    # Plot the vectorfield around the robot
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    static_visualization_of_sample_avoidance_mixed(
        robot=robot,
        n_resolution=40,
        dynamical_system=initial_dynamics,
        fast_avoider=mixed_avoider,
        plot_initial_robot=True,
        sample_environment=sampled_environment,
        # show_ticks=False,
        show_ticks=True,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        # do_quiver=True,
    )

    if save_figure:
        figure_name = "multiple_avoiding_obstacles_mixed"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def animation_with_mixed(save_animation):
    start_point = np.array([3, 4])

    # x_lim = [0, 18]
    # y_lim = [0, 8]
    x_lim = [0, 18]
    y_lim = [0, 8]

    control_radius = 0.6

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([17, 3.0]), maximum_velocity=1.0
    )

    analytic_environment = GradientContainer()
    analytic_environment.append(
        CircularObstacle(
            center_position=np.array([14, 7]),
            orientation=-20 * np.pi / 180,
            radius=0.5,
            margin_absolut=control_radius,
            linear_velocity=np.array([-0.3, -0.5]),
        )
    )

    analytic_environment.append(
        CircularObstacle(
            center_position=np.array([3, 1]),
            orientation=-20 * np.pi / 180,
            radius=0.5,
            margin_absolut=control_radius,
            # linear_velocity=np.array([-0.5, 0.2]),
        )
    )

    # Sample Environment
    sampled_environment = ShapelySamplingContainer(n_samples=50)
    # ellipse = shapely.affinity.scale(shapely.geometry.Point(14, 1).buffer(1), 2, 1)
    # ellipse = shapely.affinity.rotate(ellipse, 50)
    sampled_environment.add_obstacle(
        SampledEllipse.from_obstacle(
            position=np.array([14, 1]),
            orientation_in_degree=50,
            axes_length=np.array([2, 1]),
        )
    )

    # sampled_environment.add_obstacle(shapely.geometry.box(5, 4, 7, 8))
    sampled_environment.add_obstacle(
        SampledCuboid.from_obstacle(
            position=np.array([6, 6]),
            orientation_in_degree=00,
            axes_length=np.array([2, 4]),
        )
    )

    robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = control_radius

    robot.obstacle_environment = analytic_environment

    # Initialize Avoider
    mixed_avoider = MixedEnvironmentAvoider(
        robot=robot,
        weight_max_norm=1e9,
        weight_factor=2,
        weight_power=4.0,
        scaling_laserscan_weight=1.1,
    )

    plt.close("all")

    sampled_environment.n_samples = 100

    my_animator = MixedObstacleAnimator(
        it_max=400,
        dt_simulation=0.05,
    )

    my_animator.setup(
        robot=robot,
        initial_dynamics=initial_dynamics,
        avoider=mixed_avoider,
        environment=sampled_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        show_reference=True,
        show_ticks=False,
        show_velocity=False,
    )

    my_animator.run(save_animation=save_animation)


def scenario_mixed_analysis():
    # start_point = np.array([, 4])

    x_lim = [0, 12]
    y_lim = [0, 10]

    # x_lim = [5, 7]
    # y_lim = [5.5, 7]

    attractor_position = np.array([11.5, 9.5])

    control_radius = 0.6

    initial_dynamics = LinearSystem(
        attractor_position=attractor_position, maximum_velocity=1.0
    )

    analytic_environment = GradientContainer()
    analytic_environment.append(
        CircularObstacle(
            center_position=np.array([8, 6]),
            orientation=-20 * np.pi / 180,
            radius=0.5,
            margin_absolut=control_radius,
        )
    )

    analytic_environment.append(
        CircularObstacle(
            center_position=np.array([2.4, 5]),
            orientation=-20 * np.pi / 180,
            radius=0.5,
            margin_absolut=control_radius,
        )
    )

    sampled_environment = ShapelySamplingContainer(n_samples=50)
    # sampled_environment.create_ellipse(
    #     position=[10, 2.5], axes_length=[1.0, 0.7], orientation_in_degree=50
    #     )
    sampled_environment.add_obstacle(
        SampledEllipse.from_obstacle(
            position=np.array([10, 2.5]),
            orientation_in_degree=50,
            axes_length=np.array([1.0, 0.7]),
        )
    )

    # sampled_environment.create_cuboid(position=[4, 8], axes_length=[1.5, 0.9])
    sampled_environment.add_obstacle(
        SampledCuboid.from_obstacle(
            position=np.array([4, 8]),
            orientation_in_degree=0,
            axes_length=np.array([1.5, 0.9]),
        )
    )

    robot = QoloRobot(pose=ObjectPose(position=np.zeros(2), orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = control_radius

    robot.obstacle_environment = analytic_environment

    # Initialize Avoider
    mixed_avoider = MixedEnvironmentAvoider(
        robot=robot,
        weight_max_norm=1e6,
        delta_sampling=2 * np.pi / sampled_environment.n_samples * 5,
        # scaling_obstacle_weight=5.0,
        weight_power=1.0,
        # scaling_laserscan_weight=1.2,
    )

    # Plot the vectorfield around the robot
    # fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    fig, axs = plt.subplots(1, 2, figsize=(16, 9))

    static_visualization_of_sample_avoidance_mixed(
        robot=robot,
        n_resolution=80,
        dynamical_system=initial_dynamics,
        fast_avoider=mixed_avoider,
        # plot_initial_robot=True,
        sample_environment=sampled_environment,
        # show_ticks=False,
        show_ticks=True,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=axs[0],
        do_quiver=False,
        ax_ref=axs[1],
        plot_norm_dirs=True,
    )


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")

    # The two main pretty scenarios:
    vectorfield_with_scenario_mixed(save_figure=False)
    # animation_with_mixed(save_animation=False)

    # And then the more down-to-earth
    # scenario_mixed_analysis()
