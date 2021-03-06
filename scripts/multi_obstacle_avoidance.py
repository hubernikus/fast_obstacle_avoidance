# """ Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2021-12-19
# Email: lukas.huber@epfl.ch

import numpy as np
from numpy import linalg as LA
from numpy import pi
import matplotlib.pyplot as plt

from vartools.states import ObjectPose
from vartools.dynamical_systems import LinearSystem


from dynamic_obstacle_avoidance.obstacles import Ellipse, Sphere
from dynamic_obstacle_avoidance.obstacles import CircularObstacle
from dynamic_obstacle_avoidance.containers import ObstacleContainer, SphereContainer
from dynamic_obstacle_avoidance.visualization import plot_obstacles

# from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider
# from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider
from fast_obstacle_avoidance.control_robot import QoloRobot


def get_random_position(x_lim, y_lim):
    dimension = 2

    position = np.random.rand(dimension)
    position[0] = position[0] * (x_lim[1] - x_lim[0]) + x_lim[0]
    position[1] = position[1] * (y_lim[1] - y_lim[0]) + y_lim[0]

    return position


def double_plot(
    obstacle_environment,
    x_lim=[-5, 5],
    y_lim=[-5, 5],
    n_grid=40,
    plot_normal=True,
    attractor_position=None,
    figsize=(12, 6),
):
    if attractor_position is None:
        attractor_position = np.array([4, -0.1])

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    for ax in axs:
        ax.axis("equal")
        ax.grid(True)

        plot_obstacles(
            ax=ax,
            obstacle_container=obstacle_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            noTicks=False,
            draw_reference=True,
        )

    main_avoider = FastObstacleAvoider(
        obstacle_environment=obstacle_environment,
        reference_update_before_modulation=True,
        evaluate_velocity_weight=True,
    )

    initial_dynamics = LinearSystem(
        # attractor_position=np.array([3, -3]), maximum_velocity=0.8)
        attractor_position=attractor_position,
        maximum_velocity=0.8,
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

    # Single-Value Test
    if False:
        position = np.array([1.322, 2.320])
        initial_vel = initial_dynamics.evaluate(position=position)
        mod_vel = main_avoider.avoid(initial_velocity=initial_vel, position=position)
        print("ref dir", main_avoider.reference_direction)
        print("pos", position)
        print("vel", mod_vel)

        position = np.array([1.458, 2.317])
        initial_vel = initial_dynamics.evaluate(position=position)
        mod_vel = main_avoider.avoid(initial_velocity=initial_vel, position=position)

        print("ref dir", main_avoider.reference_direction)
        print("pos", position)
        print("vel", mod_vel)

        return

    for it in range(positions.shape[1]):
        # main_avoider.update_reference_direction(position=positions[:, it])

        initial_vel = initial_dynamics.evaluate(position=positions[:, it])

        # norm_dirs[:, it] = main_avoider.normal_direction
        # if any(relative_distances < 0):
        # continue

        # fast_avoider.update_laserscan(allscan)
        mod_vel[:, it] = main_avoider.avoid(
            initial_velocity=initial_vel, position=positions[:, it]
        )

        ref_dirs[:, it] = main_avoider.reference_direction
        if main_avoider.normal_direction is not None:
            norm_dirs[:, it] = main_avoider.normal_direction

        # print('pos', positions[:, it])
        # print('vel', LA.norm(mod_vel[:, it]), mod_vel[:, it])

        # print('ref_dirs', main_avoider.reference_direction)

    axs[0].quiver(
        positions[0, :],
        positions[1, :],
        ref_dirs[0, :],
        ref_dirs[1, :],
        color="black",
        scale=30,
        alpha=0.8,
    )

    if plot_normal:
        axs[0].quiver(
            positions[0, :],
            positions[1, :],
            norm_dirs[0, :],
            norm_dirs[1, :],
            color="#9400D3",
            scale=30,
            alpha=0.8,
        )

    axs[1].quiver(
        positions[0, :],
        positions[1, :],
        mod_vel[0, :],
        mod_vel[1, :],
        color="blue",
        scale=20,
        alpha=0.8,
    )

    axs[1].plot(
        initial_dynamics.attractor_position[0],
        initial_dynamics.attractor_position[1],
        "k*",
        linewidth=13.0,
        markersize=12,
        zorder=5,
    )

    return fig, axs


def main_vectorfield_multi_circle(
    x_lim=[-5, 5], y_lim=[-5, 5], figure_name="multi_circle_comparison"
):
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Sphere(
            radius=1.0,
            center_position=np.array([2.0, -1.6]),
            orientation=(0 * pi / 180),
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    obstacle_environment.append(
        Sphere(
            radius=1.2,
            center_position=np.array([1.2, 3.3]),
            orientation=(0 * pi / 180),
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    obstacle_environment.append(
        Sphere(
            radius=1.0,
            center_position=np.array([-2.0, 0.0]),
            orientation=(0 * pi / 180),
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    obstacle_environment.append(
        Sphere(
            radius=0.8,
            center_position=np.array([-1.0, -1.8]),
            orientation=(0 * pi / 180),
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    obstacle_environment.append(
        Sphere(
            radius=2.0,
            center_position=np.array([-4.8, -4.8]),
            orientation=(0 * pi / 180),
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    obstacle_environment.append(
        Sphere(
            radius=1.1,
            center_position=np.array([-2, 4.8]),
            orientation=(0 * pi / 180),
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    obstacle_environment.append(
        Sphere(
            radius=1.1,
            center_position=np.array([4.8, 2]),
            orientation=(0 * pi / 180),
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    double_plot(obstacle_environment, x_lim=x_lim, y_lim=y_lim, n_grid=30)

    plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def main_vectorfield_starshaped(
    x_lim=[-5, 5],
    y_lim=[-5, 5],
    figure_name="vectorfield_starshaped",
):
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Ellipse(
            axes_length=[1.0, 2.0],
            center_position=np.array([2.0, 0.0]),
            orientation=(30 * pi / 180),
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    obstacle_environment.append(
        Ellipse(
            axes_length=[1.1, 1.3],
            center_position=np.array([-2.0, 1.0]),
            orientation=(10 * pi / 180),
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    obstacle_environment.append(
        Ellipse(
            axes_length=[0.8, 1.5],
            center_position=np.array([-2.0, -2.0]),
            orientation=(60 * pi / 180),
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )
    obstacle_environment[-1].set_reference_point(
        position=np.array([-1.33, -2.69]),
        in_global_frame=True,
    )

    obstacle_environment.append(
        Ellipse(
            axes_length=[0.8, 1.5],
            center_position=np.array([-2.0, -3.5]),
            orientation=(-60 * pi / 180),
            tail_effect=False,
            repulsion_coeff=1.4,
        )
    )

    obstacle_environment[-1].set_reference_point(
        position=np.array([-1.33, -2.69]),
        in_global_frame=True,
    )

    attractor_position = np.array([4, -0.1])

    fig, axs = double_plot(
        obstacle_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=30,
        attractor_position=attractor_position,
        # plot_normal=True
    )

    nx = ny = 20
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    main_avoider = FastObstacleAvoider(obstacle_environment=obstacle_environment)
    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))
    deviation = np.zeros((positions.shape[1]))

    # DEBUG
    if False:
        pos = np.array([-2.66666667, 4.44444444])
        # vel 0.0708717999105947 [-0.03646324  0.06077207]        # position = np.array([0, 0])
        # position = np.array([2.932, 0.858])
        main_avoider.update_reference_direction(position=position)
        initial_dynamics = LinearSystem(
            # attractor_position=np.array([3, -3]), maximum_velocity=0.8)
            attractor_position=attractor_position,
            maximum_velocity=0.8,
        )

        initial_vel = initial_dynamics.evaluate(position=position)

        mod_vel = main_avoider.avoid(initial_vel)
        breakpoint()

    # if False:
    for it in range(positions.shape[1]):
        main_avoider.update_reference_direction(position=positions[:, it])

        ref_dirs = main_avoider.reference_direction

        if main_avoider.normal_direction is not None:
            norm_dirs = main_avoider.normal_direction

        deviation[it] = np.arcsin(
            np.cross(ref_dirs / LA.norm(ref_dirs), norm_dirs / LA.norm(norm_dirs))
        )

    if False:
        pcm = axs[0].contourf(
            x_vals,
            y_vals,
            deviation.reshape(nx, ny),
            # cmap='PiYG',
            cmap="bwr",
            vmin=-np.pi / 2,
            vmax=np.pi / 2,
            zorder=-3,
            alpha=0.9,
            levels=101,
        )

        cbar = fig.colorbar(
            pcm,
            ax=axs[0],
            fraction=0.035,
            ticks=[-np.pi / 4, 0, np.pi / 4],
            # ticks=[-1.0, 0, 1.0],
            # ticks=[-0.5, 0, 0.5],
            extend="neither",
        )
        cbar.ax.set_yticklabels([r"$-\frac{\pi}{4}$", "0", r"$\frac{\pi}{4}$"])

    # cbar.set_label(r"sin${}^{-1}(\mathbf{r} \times \mathbf{n})$")
    # cbar.ax.set_title(r"sin${}^{-1}(\mathbf{r} \times \mathbf{n})$", loc='left')
    # cbar.ax.set_title(r"$\mathbf{r} \times \mathbf{n}$", loc='left')
    # cbar.set_label("a", loc='left')
    # cbar.ax.set_xlabel("a")

    initial_dynamics = LinearSystem(
        # attractor_position=np.array([3, -3]), maximum_velocity=0.8)
        attractor_position=attractor_position,
        maximum_velocity=0.8,
    )

    main_avoider = FastObstacleAvoider(obstacle_environment=obstacle_environment)

    plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")
    print("Safisave")


def main_vectorfield_multi_ellipse(
    # x_lim=[-2, 2], y_lim=[-2, 2],
    x_lim=[-1, 0],
    y_lim=[-1, 0],
    figure_name="multi_ellipse_comparison",
):
    n_ellipse = 10

    obstacle_environment = ObstacleContainer()
    for ii in range(n_ellipse):
        delta_ang = ii * 2 * np.pi / n_ellipse

        pos = np.array([np.cos(delta_ang), np.sin(delta_ang)])

        obstacle_environment.append(
            Ellipse(
                center_position=pos,
                axes_length=np.array([0.6, 0.12]),
                orientation=(delta_ang + 50 * pi / 180),
                tail_effect=False,
            )
        )

        obstacle_environment[-1].set_reference_point(
            np.array([-obstacle_environment[-1].axes_length[0] * 0.8, 0]),
            in_global_frame=False,
        )

    double_plot(
        obstacle_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=30,
        attractor_position=np.array([0, 0]),
    )

    plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def main_vectorfield_single_ellipse(
    x_lim=[-2, 2], y_lim=[-2, 2], figure_name="single_ellipse_with_reference"
):
    obstacle_environment = ObstacleContainer()
    obstacle_environment.append(
        Ellipse(
            center_position=np.array([-0, 0.0]),
            axes_length=np.array([1.0, 0.4]),
            orientation=(40 * pi / 180),
            tail_effect=False,
        )
    )
    obstacle_environment[-1].set_reference_point(
        np.array([-obstacle_environment[-1].axes_length[0] * 0.8, 0]),
        in_global_frame=False,
    )

    double_plot(
        obstacle_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=30,
        attractor_position=np.array([1, 0.4]),
    )

    plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def main_multiobstacle_unifrom_circular(x_lim=[-8, 8], y_lim=[-8, 8]):
    obs_environment = SphereContainer()
    human_radius = 0.7

    robot = QoloRobot(pose=ObjectPose(position=[0, 0], orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.4

    margin = (human_radius + robot.control_radius) * 2.0
    x_lim_obs = [x_lim[0] + margin, x_lim[1] - margin]
    y_lim_obs = [y_lim[0] + margin, y_lim[1] - margin]

    np.random.seed(4)
    n_humans = 6
    for ii in range(n_humans):
        # Random ellipse
        # position = get_random_position(x_lim=x_lim, y_lim=y_lim)
        position = get_random_position(x_lim=x_lim_obs, y_lim=y_lim_obs)

        obs_environment.append(
            CircularObstacle(
                center_position=position,
                radius=human_radius,
                margin_absolut=robot.control_radius,
            )
        )

    double_plot(
        obs_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=30,
        attractor_position=np.array([4, 4]),
        figsize=(18, 9),
        plot_normal=False,
    )


def main_single_obstacle(
    x_lim=[-8, 8],
    y_lim=[-8, 8]
    # x_lim=[1, 3], y_lim=[1, 3]
):
    obs_environment = SphereContainer()
    human_radius = 0.7

    robot = QoloRobot(pose=ObjectPose(position=[0, 0], orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.4

    margin = (human_radius + robot.control_radius) * 2.0
    x_lim_obs = [x_lim[0] + margin, x_lim[1] - margin]
    y_lim_obs = [y_lim[0] + margin, y_lim[1] - margin]

    n_humans = 1
    obs_environment.append(
        CircularObstacle(
            center_position=np.array([0, 0]),
            radius=human_radius,
            margin_absolut=robot.control_radius,
        )
    )

    double_plot(
        obs_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        n_grid=30,
        attractor_position=np.array([4, 4]),
        figsize=(18, 9),
        plot_normal=False,
    )


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # main_vectorfield_multi_circle()
    # main_vectorfield_starshaped()
    # main_vectorfield_multi_ellipse()
    # main_vectorfield_single_ellipse()
    main_multiobstacle_unifrom_circular()
    # main_single_obstacle()
