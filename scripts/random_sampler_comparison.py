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
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving

from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider

from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer
from fast_obstacle_avoidance.sampling_container import visualize_obstacles

from fast_obstacle_avoidance.visualization import FastObstacleAnimator


def single_sample_run(
    avoider, dimensions, num_samples, initial_velocity, stretching_factor=1000
):
    it_repeat = 0

    data = np.random.randn(dimensions, num_samples) * stretching_factor

    # Make sure all are outisde of robot
    dist = LA.norm(data, axis=0)
    ind_close = dist < avoider.robot.control_radius
    if np.sum(ind_close):
        try:
            data[:, ind_close] = data[:, ind_close] + np.ones(
                (dimensions, np.sum(ind_close))
            )
        except:
            breakpoint()

    t_start = timer()
    avoider.update_reference_direction(data, in_robot_frame=False)
    modulated_velocity = avoider.avoid(initial_velocity)
    t_stop = timer()

    return t_stop - t_start


def comparison_dimensions_sampler(save_figure=False):
    dimensions = np.array([2, 3, 6, 7, 10, 16, 32]).astype(int)

    num_runs = 100
    sample_range = [10, 100000]

    # samples_number = np.rint(np.linspace(sample_range[0], sample_range[1], num_runs)).astype(int)
    sample_log = np.log10(sample_range)
    samples_number = np.rint(
        np.logspace(sample_log[0], sample_log[1], num_runs)
    ).astype(int)

    experiment_duration_grid = np.zeros((dimensions.shape[0], samples_number.shape[0]))

    n_repeat = 30

    for it_d, dim in enumerate(dimensions):
        robot = QoloRobot(pose=ObjectPose(position=np.zeros(dim)))
        robot.control_point = [0, 0]
        robot.control_radius = 1.0

        fast_avoider = SampledAvoider(
            robot=robot,
            weight_max_norm=1e8,
            weight_factor=5,
            weight_power=1.0,
        )

        for it_s, n_samp in enumerate(samples_number):
            inital_velocity = np.ones(dim) / dim

            duration = np.zeros(n_repeat)
            for it_r in range(n_repeat):
                duration[it_r] = single_sample_run(
                    fast_avoider, dim, n_samp, inital_velocity
                )

            experiment_duration_grid[it_d, it_s] = np.mean(duration)

    # Transform to [ms]
    experiment_duration_grid = experiment_duration_grid * 1000

    fig, ax = plt.subplots(figsize=(4, 3))

    for ii, dim in enumerate(dimensions):
        ax.plot(samples_number, experiment_duration_grid[ii, :], label=f"d={dim}")

    ax.set_xlabel("Number of datapoints")
    ax.set_ylabel("Time [ms]")

    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.grid(True)
    ax.legend()

    # breakpoint()

    if save_figure:
        figure_name = "comparison_sampling_dimensions"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")
    return experiment_duration_grid, samples_number, dimensions


def create_obstacle_container(
    dim, n_obstacles, ax_range=np.array([0.5, 2]), pos_scaling=10
):
    my_container = ObstacleContainer()

    for ii in range(n_obstacles):
        axes = np.random.rand(dim) * (ax_range[1] - ax_range[0]) + ax_range[0]

        position = np.random.rand(dim) * pos_scaling

        if LA.norm(position) < np.max(axes):
            position = position + np.ones(dim) * np.max(axes)

        my_container.append(Ellipse(center_position=position, axes_length=axes))

    return my_container


def do_multiple_modulation_avoidance(initial_velocity, container, position):
    start_time = timer()

    modulated_velocity = obs_avoidance_interpolation_moving(
        position, initial_velocity, container
    )

    end_time = timer()
    return end_time - start_time


def do_single_modulation_avoidance(initial_velocity, avoider, position):
    start_time = timer()

    avoider.update_reference_direction(position=position)
    modulated_velocity = fast_avoider.avoid(initial_velocity)

    end_time = timer()
    return end_time - start_time


def comparsion_single_mod_to_multiple():
    num_models = 2
    n_runs = 10
    sample_range = [1, 10]

    # sample_log = np.log10(sample_range)
    # samples_number = np.rint(np.logspace(sample_log[0], sample_log[1], num_runs)).astype(int)

    samples_number = np.linspace(sample_range[0], sample_range[1], num_runs).astype(int)
    experiment_duration_grid = np.zeros((num_models, samples_number.shape[0]))

    n_repeat = 30

    margin = 0.1

    robot.control_point = [0, 0]
    robot.control_radius = 1.0

    dim = 2

    it_model = 0
    for it_s, n_samp in enumerate(samples_number):
        inital_velocity = np.ones(dim) / dim

        robot = QoloRobot(pose=ObjectPose(position=np.zeros(dim)))
        robot.control_point = [0, 0]
        robot.control_radius = 1.0

        container = create_obstacle_container(dim, n_obstacles=it_s)
        fast_avoider = FastObstacleAvoider(
            robot=robot,
            weight_max_norm=1e8,
            weight_factor=5,
            weight_power=1.0,
        )

        duration = np.zeros(n_repeat)
        for it_r in range(n_repeat):
            duration[it_r] = single_fast_run(fast_avoider, dim, n_samp, inital_velocity)

            experiment_duration_grid[it_model, it_s] = np.mean(duration)

    it_model = 1
    for it_s, n_samp in enumerate(samples_number):
        inital_velocity = np.ones(dim) / dim

        robot = QoloRobot(pose=ObjectPose(position=np.zeros(dim)))
        robot.control_point = [0, 0]
        robot.control_radius = 1.0

        fast_avoider = FastObstacleAvoider(
            robot=robot,
            weight_max_norm=1e8,
            weight_factor=5,
            weight_power=1.0,
        )

        duration = np.zeros(n_repeat)
        for it_r in range(n_repeat):
            duration[it_r] = single_fast_run(fast_avoider, dim, n_samp, inital_velocity)

            experiment_duration_grid[it_model, it_s] = np.mean(duration)


if (__name__) == "__main__":
    plt.ion()
    # data = comparison_dimensions_sampler(save_figure=True)
    comparsion_single_mod_to_multiple()
