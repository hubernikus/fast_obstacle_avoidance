""" Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2021-12-19
# Email: lukas.huber@epfl.ch

import numpy as np
from numpy import linalg as LA
from numpy import pi
import matplotlib.pyplot as plt

from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.containers import ObstacleContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider


def double_plot(
    obstacle_environment, x_lim=[-5, 5], y_lim=[-5, 5],
    plot_normal=False):
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    for ax in axs:
        plot_obstacles(
            ax=ax, obs=obstacle_environment,
            x_lim=x_lim, y_lim=y_lim)

    main_avoider = FastObstacleAvoider(obstacle_environment=obstacle_environment)

    initial_dynamics = LinearSystem(
        # attractor_position=np.array([3, -3]), maximum_velocity=0.8)
        attractor_position=np.array([4, -0.1]), maximum_velocity=0.8)
        
    nx = ny = 30
    x_vals, y_vals = np.meshgrid(np.linspace(x_lim[0], x_lim[1], nx),
                                 np.linspace(y_lim[0], y_lim[1], ny))
    
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    velocities = np.zeros(positions.shape)
    mod_vel = np.zeros(positions.shape)

    norm_dirs = np.zeros(positions.shape)
    ref_dirs = np.zeros(positions.shape)
    
    for it in range(positions.shape[1]):
        main_avoider.update_normal_direction(positions[:, it])

        initial_vel = initial_dynamics.evaluate(positions[:, it])
        
        ref_dirs[:, it] = main_avoider.reference_direction
        if main_avoider.normal_direction is not None:
            norm_dirs[:, it] = main_avoider.normal_direction
            
        # norm_dirs[:, it] = main_avoider.normal_direction
        # if any(relative_distances < 0):
            # continue
            
        # fast_avoider.update_laserscan(allscan)
        mod_vel[:, it] = main_avoider.avoid(initial_vel)


    axs[0].quiver(positions[0, :], positions[1, :],
                  ref_dirs[0, :], ref_dirs[1, :],
                  color="red", scale=30, alpha=0.8)

    if plot_normal:
        axs[0].quiver(positions[0, :], positions[1, :],
                      norm_dirs[0, :], norm_dirs[1, :],
                      color='#9400D3',
                      scale=30, alpha=0.8)

    axs[1].quiver(positions[0, :], positions[1, :],
                  mod_vel[0, :], mod_vel[1, :],
                  color="blue", scale=30, alpha=0.8)

    axs[1].plot(
        initial_dynamics.attractor_position[0],
        initial_dynamics.attractor_position[1],
        "k*",
        markeredgewidth=1,
        markersize=8,
        zorder=5,
        )


    return fig, ax

    
    
def main_vectorfield_two_circle(
    x_lim=[-5, 5], y_lim=[-5, 5], figure_name="two_circle_comparison"):
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Ellipse(
        axes_length=[1.0, 1.0],
        center_position=np.array([2.0, 0.0]),
        orientation=(0 * pi / 180),
        tail_effect=False,
        repulsion_coeff=1.4,
        )
    )
    
    obstacle_environment.append(
        Ellipse(
        axes_length=[1.0, 1.0],
        center_position=np.array([-2.0, 0.0]),
        orientation=(0 * pi / 180),
        tail_effect=False,
        repulsion_coeff=1.4,
        )
    )

    double_plot(obstacle_environment, x_lim=x_lim, y_lim=y_lim)

    plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")



def main_vectorfield_two_ellipse(
    x_lim=[-5, 5], y_lim=[-5, 5], figure_name="two_ellipse_comparison"):
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Ellipse(
        axes_length=[1.0, 2.0],
        center_position=np.array([2.0, 0.0]),
        orientation=(0 * pi / 180),
        tail_effect=False,
        repulsion_coeff=1.4,
        )
    )
    
    obstacle_environment.append(
        Ellipse(
        axes_length=[1.0, 2.0],
        center_position=np.array([-2.0, 0.0]),
        orientation=(0 * pi / 180),
        tail_effect=False,
        repulsion_coeff=1.4,
        )
    )

    double_plot(obstacle_environment, x_lim=x_lim, y_lim=y_lim, plot_normal=True)

    initial_dynamics = LinearSystem(
        # attractor_position=np.array([3, -3]), maximum_velocity=0.8)
        attractor_position=np.array([4, -0.1]), maximum_velocity=0.8)

    main_avoider = FastObstacleAvoider(obstacle_environment=obstacle_environment)

    pos = np.array([-3.6, -2.6])
    main_avoider.update_normal_direction(pos)

    initial_vel = initial_dynamics.evaluate(pos)
    mod_vel = main_avoider.avoid(initial_vel)
        
    ref_dirs = main_avoider.reference_direction
    if main_avoider.normal_direction is not None:
        norm_dirs = main_avoider.normal_direction
    

    plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    # plt.close('all')
    plt.ion()

    # main_vectorfield_two_circle()
    main_vectorfield_two_ellipse()
