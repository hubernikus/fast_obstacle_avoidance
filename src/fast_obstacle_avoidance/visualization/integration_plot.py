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

from dynamic_obstacle_avoidance.obstacles import CircularObstacle
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider
from fast_obstacle_avoidance.obstacle_avoider import MixedEnvironmentAvoider

from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer
from fast_obstacle_avoidance.sampling_container import visualize_obstacles

from fast_obstacle_avoidance.visualization import LaserscanAnimator
from fast_obstacle_avoidance.visualization import FastObstacleAnimator
from fast_obstacle_avoidance.visualization import MixedObstacleAnimator


def visualization_mixed_environment_with_multiple_integration(
    dynamical_system,
    start_positions,
    delta_time=0.1,
    max_it=1000,
    sample_environment=None,
    fast_avoider=None,
    robot=None,
    show_ticks=False,
    plot_initial_robot=False,
    x_lim=None,
    y_lim=None,
    ax=None,
):
    """Visualization of mixed environment."""
    # Created for 2D
    dimension = 2

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

    if plot_initial_robot:
        robot.plot2D(ax=ax)

        ax.plot(
            robot.pose.position[0],
            robot.pose.position[1],
            "o",
            color="black",
            markersize=13,
            zorder=5,
        )

        if sample_environment is not None:
            data_points = sample_environment.get_surface_points(
                center_position=robot.pose.position,
            )

            ax.plot(data_points[0, :], data_points[1, :], "o", color="k")

            fast_avoider.update_laserscan(data_points, in_robot_frame=False)

        # Store all
        # initial_velocity = dynamical_system.evaluate(robot.pose.position)
        # modulated_velocity = fast_avoider.avoid(initial_velocity)

    trajectories = []
    velocities_init = []
    velocities_mod = []

    for it_traj in range(start_positions.shape[1]):
        robot.pose.position = start_positions[:, it_traj]

        trajectories.append(np.zeros((dimension, max_it)))
        velocities_init.append(np.zeros((dimension, max_it)))
        velocities_mod.append(np.zeros((dimension, max_it)))
        trajectories[-1][:, 0] = robot.pose.position

        for ii in range(max_it - 1):
            if hasattr(fast_avoider, "obstacle_environment"):
                is_inside_an_obstacle = False
                for obs in fast_avoider.obstacle_environment:
                    if obs.get_gamma(robot.pose.position, in_global_frame=True) < 1:
                        is_inside_an_obstacle = True
                        break

                if is_inside_an_obstacle:
                    break

            if sample_environment is not None:
                if sample_environment.is_inside(
                    position=robot.pose.position, margin=robot.control_radius
                ):
                    break

                data_points = sample_environment.get_surface_points(
                    center_position=robot.pose.position,
                )

                fast_avoider.update_laserscan(data_points, in_robot_frame=False)

            velocities_init[-1][:, ii] = dynamical_system.evaluate(robot.pose.position)
            velocities_mod[-1][:, ii] = fast_avoider.avoid(velocities_init[-1][:, ii])

            robot.pose.position = (
                robot.pose.position + delta_time * velocities_mod[-1][:, ii]
            )
            trajectories[-1][:, ii + 1] = robot.pose.position

        trajectories[-1] = trajectories[-1][:, : ii + 1]
        velocities_init[-1] = velocities_init[-1][:, :ii]
        velocities_mod[-1] = velocities_mod[-1][:, :ii]

    # Plot trajectories
    for traj in trajectories:
        traj_plt = ax.plot(traj[0, :], traj[1, :])
        traj_clr = traj_plt[0].get_color()
        ax.plot(traj[0, 0], traj[1, 0], "o", color=traj_clr)
        # ax.plot(traj[0, -1], traj[1, -1], 'x', color=traj_clr)

    if sample_environment is not None:
        visualize_obstacles(sample_environment, ax=ax)

    if hasattr(fast_avoider, "obstacle_environment"):
        plot_obstacles(
            ax=ax,
            obstacle_container=fast_avoider.obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            drawVelArrow=False,
        )

    if hasattr(dynamical_system, "attractor_position"):
        ax.plot(
            dynamical_system.attractor_position[0],
            dynamical_system.attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if not show_ticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    ax.set_aspect("equal")
    # ax.grid(True)

    return ax
