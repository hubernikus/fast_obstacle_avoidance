# """ Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2021-12-19
# Email: lukas.huber@epfl.ch

import numpy as np
from numpy import linalg as LA
from numpy import pi
import matplotlib.pyplot as plt

from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.avoidance import obs_avoidance_interpolation_moving

from dynamic_obstacle_avoidance.obstacles import Ellipse, Sphere
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider
from fast_obstacle_avoidance.control_robot import QoloRobot


def single_cirlce_in_corner(
    x_lim=[-1.0, 8.0],
    y_lim=[-2.0, 4.5],
    n_grid=120,
    plot_normal=True,
    attractor_position=None,
):

    fig, ax = plt.subplots(1, 1, figsize=(11, 7))

    initial_dynamics = LinearSystem(
        attractor_position=np.array([-4, -5]),
        maximum_velocity=0.8,
    )
    
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(Ellipse(
        axes_length=np.array([2, 2]),
        center_position=np.array([0, 0])
        ))

    main_avoider = FastObstacleAvoider(
        obstacle_environment=obstacle_environment
        )
    

    plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            noTicks=True,
            draw_reference=True,
        )

    nx = ny = n_grid
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities = np.zeros(positions.shape)
    mod_vel = np.zeros(positions.shape)

    norm_dirs = np.zeros(positions.shape)
    ref_dirs = np.zeros(positions.shape)

    for it in range(positions.shape[1]):
        if LA.norm(positions[:, it]) < obstacle_environment[0].axes_length[0]:
            continue
        
        main_avoider.update_reference_direction(position=positions[:, it])
        initial_vel = initial_dynamics.evaluate(position=positions[:, it])
        
        # mod_vel[:, it] = main_avoider.avoid(initial_vel)
        # initial_velds_init = (position, x0=xAttractor)
        mod_vel[:, it] = obs_avoidance_interpolation_moving(
            positions[:, it], initial_vel, obstacle_environment)

    # ax.quiver(
        # positions[0, :],
        # positions[1, :],
        # mod_vel[0, :],
        # mod_vel[1, :],
        # color="black",
        # scale=30,
        # alpha=0.8,
    # )
    
    # if True:
        # return
    ax.streamplot(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        mod_vel[0, :].reshape(nx, ny),
        mod_vel[1, :].reshape(nx, ny),
        color="black",
        # scale=30,
        # alpha=1.0,
    )

    figure_name = "avoidance_circle"
    plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")

if (__name__) == "__main__":
    # plt.close('all')
    plt.ion()

    single_cirlce_in_corner()
