""" Script to evaluate the rosbag. """
# from timeit import default_timer as timer
import sys

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from vartools.states import ObjectPose
from vartools import dynamical_systems

from dynamic_obstacle_avoidance import containers
from dynamic_obstacle_avoidance import obstacles
from dynamic_obstacle_avoidance.visualization import plot_obstacles
from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving

from fast_obstacle_avoidance.control_robot import QoloRobot
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider


def get_multigamma(obstacle_environment, position):
    min_gamma = 1e9  # large value
    for obs in obstacle_environment:
        new_gamma = obs.get_gamma(position, in_global_frame=True)
        min_gamma = np.minimum(new_gamma, min_gamma)

    return min_gamma


def run_vectorfield_with_many_obstacles():
    x_lim = [-3.5, 3.5]
    y_lim = [-5, 5]

    radius_human = 0.3
    margin = 0.15

    dimension = 2

    radius_with_margin = radius_human + margin

    num_obs = 20
    position_list = np.zeros((dimension, num_obs))

    it_safety = 0
    ii = 0
    while ii < num_obs:
        it_safety += 1
        if it_safety > 10000:
            print("[WARNING] Exited without initializing all obstacles.")
            break

        new_pos = np.random.rand(2)
        new_pos[0] = new_pos[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
        new_pos[1] = new_pos[1] * (y_lim[1] - y_lim[0]) + y_lim[0]

        if any(
            LA.norm(position_list[:, :ii] - np.tile(new_pos, (ii, 1)).T, axis=0)
            < 2 * radius_human
        ):
            continue

        position_list[:, ii] = new_pos
        ii += 1

    # obstacles.Shpere()

    obstacle_environment = containers.ObstacleContainer()

    for ii in range(position_list.shape[1]):
        obstacle_environment.append(
            obstacles.Sphere(
                center_position=position_list[:, ii],
                radius=radius_human,
                margin_absolut=margin,
            )
        )

    # obstacle_environment.update_reference_points()

    # Same resolution x & y
    nx = 80
    dx = (x_lim[1] - x_lim[0]) / nx
    ny = int((y_lim[1] - y_lim[0]) / dx)

    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    my_dynamics = dynamical_systems.LinearSystem(
        attractor_position=np.array([0, -4]),
        maximum_velocity=1,
        distance_decrease=0.3,
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities_modu = np.zeros(positions.shape)
    velocities_init = np.zeros(positions.shape)

    multi_gamma = np.zeros(positions.shape[1])

    for it in range(positions.shape[1]):
        multi_gamma[it] = get_multigamma(obstacle_environment, positions[:, it])
        if multi_gamma[it] < 1:
            continue

        velocities_init[:, it] = my_dynamics.evaluate(positions[:, it])
        velocities_modu[:, it] = obs_avoidance_interpolation_moving(
            positions[:, it],
            velocities_init[:, it],
            obstacle_environment,
        )

    fig, ax = plt.subplots(figsize=(7, 10))
    plot_obstacles(
        ax,
        obstacle_environment,
        x_lim,
        y_lim,
        pos_attractor=my_dynamics.attractor_position,
    )

    ax.set_aspect("equal", "box")
    # axis('equal', 'box')
    # ax.axes('equal')

    # ax.quiver(positions[0, :], positions[1, :],
    # velocities_modu[0, :], velocities_modu[1, :], color="black")

    ax.streamplot(
        x_vals,
        y_vals,
        velocities_modu[0, :].reshape(ny, nx),
        velocities_modu[1, :].reshape(ny, nx),
        color="black",
    )

    ax.contourf(
        x_vals,
        y_vals,
        multi_gamma.reshape(ny, nx),
        # cmap='hot'
        cmap="GnBu",
    )

    ax.tick_params(
        axis="both",
        which="major",
        labelbottom=False,
        labelleft=False,
        bottom=False,
        top=False,
        left=False,
        right=False,
    )

    plt.ion()
    plt.show()

    fig_name = "crowd_movement"
    if True:
        plt.savefig("figures/" + fig_name + ".png", bbox_inches="tight", dpi=1000)


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    np.random.seed(42)
    run_vectorfield_with_many_obstacles()
