""" Script to create plots. """
# Author: Lukas Huber
# Created: 2023-04-01
# Email: lukas.huber@epfl.ch

from attrs import define, field

import numpy as np
import matplotlib.pyplot as plt

from vartools.states import ObjectPose
from vartools.animator import Animator
from vartools.dynamical_systems import LinearSystem

from fast_obstacle_avoidance.sampling_container import SampledCuboid
from fast_obstacle_avoidance.obstacle_avoider import SampledClusterAvoider
from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer

from fast_obstacle_avoidance.visualization import LaserscanAnimator

from fast_obstacle_avoidance.sampling_container import visualize_obstacles


@define
class AvoidanceController:
    avoider: SampledClusterAvoider
    dynamics: LinearSystem
    environment: ShapelySamplingContainer


def trajectory_integration(
    position_start,
    controller,
    time_step: float = 0.1,
    it_max: int = 100,
    convergence_error: float = 1e-2,
):
    positions = np.zeros((position_start.shape[0], it_max + 1))
    positions[:, 0] = position_start

    for ii in range(it_max):
        initial_velocity = controller.dynamics.evaluate(positions[:, ii])

        # Retrieve data-points from sampler (cartesian representation)
        datapoints = controller.environment.get_surface_points(
            center_position=positions[:, ii],
            null_direction=initial_velocity,
        )
        controller.avoider.update_laserscan(datapoints, in_robot_frame=False)

        # Modulate initial velocity
        modulated_velocity = controller.avoider.avoid(
            initial_velocity, positions[:, ii]
        )

        if np.linalg.norm(modulated_velocity) < convergence_error:
            print(f"Convergence at it={ii}")
            return positions[:, : ii + 1]

        positions[:, ii + 1] = positions[:, ii] + modulated_velocity * time_step

    return positions


def comparison_weight_factor(n_points, save_figure=True):
    x_lim = [-6, 6]
    y_lim = [-5, 5]

    start_positions = np.vstack(
        (
            np.ones(n_points) * x_lim[0],
            np.linspace(y_lim[0], y_lim[1], n_points),
        )
    )

    environment = ShapelySamplingContainer(n_samples=100)
    environment.add_obstacle(
        # SampledEllipse.from_obstacle(
        SampledCuboid.from_obstacle(
            position=np.array([0.0, 0.0]),
            orientation_in_degree=00,
            axes_length=np.array([2.0, 2.0]),
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([5.0, 0.0]), maximum_velocity=1.0
    )

    robot = QoloRobot(pose=ObjectPose(position=np.zeros(2), orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.4

    # Setup avoider + parameters
    fast_avoider = SampledClusterAvoider(control_radius=robot.control_radius)

    fast_avoider.weight_power = 1.0 / 2
    fast_avoider.weight_max_norm = 1e7

    controller = AvoidanceController(
        avoider=fast_avoider,
        dynamics=initial_dynamics,
        environment=environment,
    )

    ii = 1  # TODO: for loop
    weight_factors = [1, 3, 10]
    for ii, weight_factor in enumerate(weight_factors):

        fast_avoider.weight_factor = (
            2 * np.pi / environment.n_samples * weight_factors[ii]
        )

        fig, ax = plt.subplots(figsize=(4, 3))
        visualize_obstacles(container=environment, ax=ax, x_lim=x_lim, y_lim=y_lim)

        ax.plot(
            controller.dynamics.attractor_position[0],
            controller.dynamics.attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

        for jj in range(start_positions.shape[1]):
            # robot.pose.position = start_positions[:, ii]

            positions = trajectory_integration(
                start_positions[:, jj], controller=controller, it_max=300
            )

            ax.plot(positions[0, :], positions[1, :], color="blue")
            ax.plot(positions[0, 0], positions[1, 0], "o", color="black")

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_aspect("equal")

        if save_figure:
            figname = f"comparison_weight_factor_{weight_factor}"
            plt.savefig(
                "figures/" + figname + figtype,
                bbox_inches="tight",
            )


def comparison_weight_power(n_points, save_figure=True):
    x_lim = [-6, 6]
    y_lim = [-5, 5]

    start_positions = np.vstack(
        (
            np.ones(n_points) * x_lim[0],
            np.linspace(y_lim[0], y_lim[1], n_points),
        )
    )

    environment = ShapelySamplingContainer(n_samples=100)
    environment.add_obstacle(
        # SampledEllipse.from_obstacle(
        SampledCuboid.from_obstacle(
            position=np.array([0.0, 0.0]),
            orientation_in_degree=00,
            axes_length=np.array([2.0, 2.0]),
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([5.0, 0.0]), maximum_velocity=1.0
    )

    robot = QoloRobot(pose=ObjectPose(position=np.zeros(2), orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.4

    # Setup avoider + parameters
    fast_avoider = SampledClusterAvoider(control_radius=robot.control_radius)

    # fast_avoider.weight_power = 1.0 / 2
    fast_avoider.weight_max_norm = 1e7
    fast_avoider.weight_factor = 2 * np.pi / environment.n_samples * 3

    controller = AvoidanceController(
        avoider=fast_avoider,
        dynamics=initial_dynamics,
        environment=environment,
    )

    ii = 1  # TODO: for loop
    # weight_factors = [0.1, 1, 10]
    weight_powers = [1.0, 1.5, 2.0]
    for ii, weight_power in enumerate(weight_powers):
        fast_avoider.weight_power = weight_power

        fig, ax = plt.subplots(figsize=(4, 3))
        visualize_obstacles(container=environment, ax=ax, x_lim=x_lim, y_lim=y_lim)

        ax.plot(
            controller.dynamics.attractor_position[0],
            controller.dynamics.attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

        for jj in range(start_positions.shape[1]):
            # robot.pose.position = start_positions[:, ii]

            positions = trajectory_integration(
                start_positions[:, jj], controller=controller, it_max=200
            )

            ax.plot(positions[0, :], positions[1, :], color="blue")
            ax.plot(positions[0, 0], positions[1, 0], "o", color="black")

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_aspect("equal")

        if save_figure:
            figname = f"comparison_weight_power_{weight_power}"
            plt.savefig(
                "figures/" + figname + figtype,
                bbox_inches="tight",
            )


def comparison_weight_max_norm(n_points, save_figure=True):
    x_lim = [-4, 4]
    y_lim = [-3, 3]

    start_positions = np.vstack(
        (
            np.ones(n_points) * x_lim[0],
            np.linspace(y_lim[0], y_lim[1], n_points),
        )
    )

    environment = ShapelySamplingContainer(n_samples=100)
    environment.add_obstacle(
        # SampledEllipse.from_obstacle(
        SampledCuboid.from_obstacle(
            position=np.array([0.0, 0.0]),
            orientation_in_degree=00,
            axes_length=np.array([2.0, 2.0]),
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([5.0, 0.0]), maximum_velocity=1.0
    )

    robot = QoloRobot(pose=ObjectPose(position=np.zeros(2), orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.4

    # Setup avoider + parameters
    fast_avoider = SampledClusterAvoider(control_radius=robot.control_radius)

    fast_avoider.weight_power = 1.5
    # fast_avoider.weight_max_norm = 1e7
    fast_avoider.weight_factor = 2 * np.pi / environment.n_samples * 3

    controller = AvoidanceController(
        avoider=fast_avoider,
        dynamics=initial_dynamics,
        environment=environment,
    )

    weight_max_norms = [1e1, 1e2, 1e3]
    for ii, weight_max_norm in enumerate(weight_max_norms):
        fast_avoider.weight_max_norm = weight_max_norm

        fig, ax = plt.subplots(figsize=(4, 3))
        visualize_obstacles(container=environment, ax=ax, x_lim=x_lim, y_lim=y_lim)

        ax.plot(
            controller.dynamics.attractor_position[0],
            controller.dynamics.attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

        for jj in range(start_positions.shape[1]):
            # robot.pose.position = start_positions[:, ii]

            positions = trajectory_integration(
                start_positions[:, jj], controller=controller, it_max=200
            )

            ax.plot(positions[0, :], positions[1, :], color="blue")
            ax.plot(positions[0, 0], positions[1, 0], "o", color="black")

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_aspect("equal")

        if save_figure:
            figname = f"comparison_weight_max_{weight_max_norm}"
            plt.savefig(
                "figures/" + figname + figtype,
                bbox_inches="tight",
            )


def comparison_weight_control_radius(n_points, save_figure=True):
    x_lim = [-4, 4]
    y_lim = [-3, 3]

    start_positions = np.vstack(
        (
            np.ones(n_points) * x_lim[0],
            np.linspace(y_lim[0], y_lim[1], n_points),
        )
    )

    environment = ShapelySamplingContainer(n_samples=100)
    environment.add_obstacle(
        # SampledEllipse.from_obstacle(
        SampledCuboid.from_obstacle(
            position=np.array([0.0, 0.0]),
            orientation_in_degree=00,
            axes_length=np.array([2.0, 2.0]),
        )
    )

    initial_dynamics = LinearSystem(
        attractor_position=np.array([5.0, 0.0]), maximum_velocity=1.0
    )

    robot = QoloRobot(pose=ObjectPose(position=np.zeros(2), orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.4

    # Setup avoider + parameters
    fast_avoider = SampledClusterAvoider(control_radius=robot.control_radius)

    fast_avoider.weight_power = 1.5
    fast_avoider.weight_max_norm = 1e7
    fast_avoider.weight_factor = 2 * np.pi / environment.n_samples * 3

    controller = AvoidanceController(
        avoider=fast_avoider,
        dynamics=initial_dynamics,
        environment=environment,
    )

    weight_max_norms = [1e1, 1e2, 1e3]
    for ii, weight_max_norm in enumerate(weight_max_norms):
        fast_avoider.weight_max_norm = weight_max_norm

        fig, ax = plt.subplots(figsize=(4, 3))
        visualize_obstacles(container=environment, ax=ax, x_lim=x_lim, y_lim=y_lim)

        ax.plot(
            controller.dynamics.attractor_position[0],
            controller.dynamics.attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

        for jj in range(start_positions.shape[1]):
            # robot.pose.position = start_positions[:, ii]

            positions = trajectory_integration(
                start_positions[:, jj], controller=controller, it_max=200
            )

            ax.plot(positions[0, :], positions[1, :], color="blue")
            ax.plot(positions[0, 0], positions[1, 0], "o", color="black")

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.set_aspect("equal")

        if save_figure:
            figname = f"comparison_weight_max_{weight_max_norm}"
            plt.savefig(
                "figures/" + figname + figtype,
                bbox_inches="tight",
            )


if (__name__) == "__main__":
    figtype = ".pdf"

    plt.ion()
    plt.close("all")

    # comparison_weight_factor(n_points=7)
    # comparison_weight_power(n_points=7)
    comparison_weight_max_norm(n_points=7)

    print("Done all.")
