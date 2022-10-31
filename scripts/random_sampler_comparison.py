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

from fast_obstacle_avoidance.comparison.vfh_avoider import VFH_Avoider


figureformat = ".png"
# figureformat = ".pdf"


@dataclass
class DataHandler:
    n_repeat: int = 30
    experiment_duration_grid = None
    experiment_duration_baseline = None

    samples_number = None
    dimensions = None

    sample_range = None


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


def comparison_dimensions_sampler(num_runs=100, n_repeat=40):
    dh = DataHandler(n_repeat=n_repeat)
    dh.dimensions = np.array([2, 3, 6, 7, 10, 16, 32]).astype(int)

    sample_range = [10, 100000]

    # samples_number = np.rint(np.linspace(sample_range[0], sample_range[1], num_runs)).astype(int)
    sample_log = np.log10(sample_range)
    dh.samples_number = np.rint(
        np.logspace(sample_log[0], sample_log[1], num_runs)
    ).astype(int)

    dh.experiment_duration_grid = np.zeros(
        (dh.dimensions.shape[0], dh.samples_number.shape[0])
    )

    for it_d, dim in enumerate(dh.dimensions):
        robot = QoloRobot(pose=ObjectPose(position=np.zeros(dim)))
        robot.control_point = [0, 0]
        robot.control_radius = 1.0

        fast_avoider = SampledAvoider(
            robot=robot,
            weight_max_norm=1e8,
            weight_factor=5,
            weight_power=1.0,
        )

        for it_s, n_samp in enumerate(dh.samples_number):
            inital_velocity = np.ones(dim) / dim

            duration = np.zeros(dh.n_repeat)
            for it_r in range(dh.n_repeat):
                duration[it_r] = single_sample_run(
                    fast_avoider, dim, n_samp, inital_velocity
                )

            dh.experiment_duration_grid[it_d, it_s] = np.mean(duration)

    return dh


def comparison_baseline(dh, dim=2):
    dh.experiment_duration_baseline = np.zeros(dh.samples_number.shape[0])

    robot = QoloRobot(pose=ObjectPose(position=np.zeros(dim)))
    robot.control_point = [0, 0]
    robot.control_radius = 1.0

    fast_avoider = VFH_Avoider(
        robot=robot,
    )

    for it_s, n_samp in enumerate(dh.samples_number):
        # Reset each time since hyper-parameters have to be set
        fast_avoider.compute_histogram_props(n_max_sections=n_samp)

        inital_velocity = np.ones(dim) / dim

        duration = np.zeros(dh.n_repeat)
        for it_r in range(dh.n_repeat):
            duration[it_r] = single_sample_run(
                fast_avoider, dim, n_samp, inital_velocity
            )

        dh.experiment_duration_baseline[it_s] = np.mean(duration)

    return dh


def plot_sampled_comparison(dh, save_figure=False):
    # Transform to [ms]
    dh.experiment_duration_grid = dh.experiment_duration_grid * 1000
    dh.experiment_duration_baseline = dh.experiment_duration_baseline * 1000
    dh.samples_number = dh.samples_number / 1000

    # Setup
    sns.set_style("darkgrid")

    # darkgrid, white grid, dark, white and ticksplt.rc('axes', titlesize=18)     # fontsize of the axes title
    plt.rc("axes", labelsize=12)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=11)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=11)  # fontsize of the tick labels
    plt.rc("legend", fontsize=12)  # legend fontsize
    plt.rc("font", size=11)  # controls default text sizes
    plt.rc("lines", linewidth=2.5)

    fig, ax = plt.subplots(figsize=(4.5, 3.0), tight_layout=True)
    # fig, ax = plt.subplots(figsize=(4.0, 3.0), tight_layout=True)

    log_mean_x = []
    mean_x_list = []
    mean_y_list = []

    # colors = sns.color_palette("pastel")
    colors = sns.color_palette("husl", dh.dimensions.shape[0])

    # fig, ax = plt.subplots(figsize=(4, 3))

    for ii, dim in enumerate(dh.dimensions):
        ax.plot(
            dh.samples_number,
            dh.experiment_duration_grid[ii, :],
            label=f"d={dim}",
            color=colors[ii],
        )

    ax.set_xlabel("Number of datapoints [1000]")
    ax.set_ylabel("Time [ms]")

    ax.set_xlim([dh.samples_number[0], dh.samples_number[-1]])
    # ax.set_ylim([0, 7.2])
    ax.set_ylim([0, 10])

    # ax.set_xscale("log")
    # ax.set_yscale("log")

    if dh.experiment_duration_baseline is not None:
        ax.plot(
            dh.samples_number,
            dh.experiment_duration_baseline,
            "--",
            label=f"VFH",
            color=colors[0],
        )

    ax.grid(True)
    ax.legend()

    if save_figure:
        figure_name = "comparison_sampling_dimensions_nolog"
        plt.savefig(
            "figures/" + figure_name + figureformat, bbox_inches="tight", dpi=1200
        )

    return fig, ax


if (__name__) == "__main__":
    plt.ion()
    # data = comparison_dimensions_sampler(save_figure=True)
    dh = comparison_dimensions_sampler(num_runs=3, n_repeat=10)

    # TODO - rerun when nothing is going on...
    # dh = comparison_dimensions_sampler()
    dh = comparison_baseline(dh)

    plot_sampled_comparison(dh, save_figure=True)
