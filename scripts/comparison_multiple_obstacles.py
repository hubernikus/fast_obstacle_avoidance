""" Script to create plots. """
# Author: Lukas Huber
# Created: 2021-02-22
# Email: lukas.huber@epfl.ch

import copy
from dataclasses import dataclass
from timeit import default_timer as timer

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt
import seaborn as sns

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


figureformat = ".png"
# figureformat = ".pdf"


@dataclass
class DataHandler:
    experiment_duration_grid = None
    samples_number = None


def create_obstacle_container(
    dim, n_obstacles, ax_range=np.array([0.5, 2]), pos_scaling=10
):
    my_container = GradientContainer()

    for ii in range(n_obstacles):
        axes = np.random.rand(dim) * (ax_range[1] - ax_range[0]) + ax_range[0]

        position = np.random.rand(dim) * pos_scaling

        if LA.norm(position) < np.max(axes):
            position = position + np.ones(dim) * np.max(axes)

        my_container.append(Ellipse(center_position=position, axes_length=axes))

    return my_container


def time_multimodulation_run(initial_velocity, container, position):
    start_time = timer()

    modulated_velocity = obs_avoidance_interpolation_moving(
        position, initial_velocity, container
    )

    end_time = timer()
    return end_time - start_time


def time_singemodulation_run(inital_velocity, avoider, position):
    start_time = timer()

    avoider.update_reference_direction(position=position)
    modulated_velocity = avoider.avoid(inital_velocity)

    end_time = timer()
    return end_time - start_time


def comparsion_single_mod_to_multiple(num_runs=20, n_repeat=30):
    num_models = 2

    sample_range = [1, 1000]

    # sample_log = np.log10(sample_range)
    # samples_number = np.rint(np.logspace(sample_log[0], sample_log[1], num_runs)).astype(int)
    # samples_number = np.linspace(sample_range[0], sample_range[1], num_runs).astype(int)
    sample_log = np.log10(sample_range)
    samples_number = np.rint(
        np.logspace(sample_log[0], sample_log[1], num_runs)
    ).astype(int)
    experiment_duration_grid = np.zeros((num_models, samples_number.shape[0]))

    dim = 2

    robot = QoloRobot(pose=ObjectPose(position=np.zeros(dim)))
    robot.control_point = [0, 0]
    robot.control_radius = 1.0

    initial_velocity = np.ones(dim) / dim
    position = np.zeros(dim)

    # robot = QoloRobot(pose=ObjectPose(position=np.zeros(dim)))
    # robot.pose.position = position
    # robot.control_point = [0, 0]
    # robot.control_radius = 1.0

    fast_avoider = FastObstacleAvoider(
        robot=robot,
        obstacle_environment=[],
        weight_max_norm=1e8,
        weight_factor=5,
        weight_power=1.0,
    )

    it_model = 0
    for it_s, n_samp in enumerate(samples_number):
        print(f"Model: {it_model}   | Sample: {n_samp}")

        duration = np.zeros(n_repeat)
        for it_r in range(n_repeat):

            fast_avoider.obstacle_environment = create_obstacle_container(
                dim, n_obstacles=it_s
            )

            duration[it_r] = time_singemodulation_run(
                initial_velocity, avoider=fast_avoider, position=position
            )

        experiment_duration_grid[it_model, it_s] = np.mean(duration)

    it_model = 1

    inital_velocity = np.ones(dim) / dim
    position = np.zeros(dim)

    for it_s, n_samp in enumerate(samples_number):
        print(f"Model: {it_model}   | Sample: {n_samp}")

        duration = np.zeros(n_repeat)
        for it_r in range(n_repeat):
            container = create_obstacle_container(dim, n_obstacles=it_s)

            duration[it_r] = time_multimodulation_run(
                initial_velocity, container=container, position=position
            )

            experiment_duration_grid[it_model, it_s] = np.mean(duration)

    dh = DataHandler()
    dh.experiment_duration_grid = experiment_duration_grid
    dh.samples_number = samples_number

    return dh


def plot_dimension_data(dh, save_figure=True):
    # Setup
    sns.set_style("darkgrid")

    # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc("axes", labelsize=12)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=11)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=11)  # fontsize of the tick labels
    plt.rc("legend", fontsize=12)  # legend fontsize
    plt.rc("font", size=11)  # controls default text sizes
    plt.rc("lines", linewidth=2.5)

    # Transform to [ms]
    experiment_duration_grid = dh.experiment_duration_grid * 1000

    # fig, ax = plt.subplots(figsize=(4, 3))
    fig, ax = plt.subplots(figsize=(4.5, 3.0), tight_layout=True)

    ax.plot(dh.samples_number, experiment_duration_grid[0, :], label=f"FOA")
    ax.plot(dh.samples_number, experiment_duration_grid[1, :], label=f"MuMo")

    ax.set_xlabel("Number of obstacles")
    ax.set_ylabel("Time [ms]")

    # ax.set_xscale("log")
    # ax.set_yscale("log")

    ax.grid(True)
    ax.legend()

    ax.set_xlim(dh.samples_number[0], dh.samples_number[-1])
    # ax.set_ylim(0, ax.get_ylim()[1])
    # ax.set_ylim([0, 7.2])
    ax.set_ylim([0, 10])  # Same limit as other comparitor

    # breakpoint()

    if save_figure:
        figure_name = "comparions_random_ellipse_number_d2"
        plt.savefig(
            "figures/" + figure_name + figureformat, bbox_inches="tight", dpi=1200
        )


if (__name__) == "__main__":
    plt.ion()
    dh = comparsion_single_mod_to_multiple()
    # dh = comparsion_single_mod_to_multiple(num_runs=3, n_repeat=3)
    plot_dimension_data(dh, save_figure=True)
