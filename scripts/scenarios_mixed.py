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

# from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider
from fast_obstacle_avoidance.obstacle_avoider import MixedEnvironmentAvoider
from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer
from fast_obstacle_avoidance.sampling_container import visualize_obstacles

from fast_obstacle_avoidance.visualization import FastObstacleAnimator
from fast_obstacle_avoidance.visualization import MixedObstacleAnimator


def static_visualization_of_sample_avoidance_mixed(
    sample_environment,
    dynamical_system,
    fast_avoider=None,
    n_resolution=30,
    robot=None,
    show_ticks=False,
    plot_initial_robot=False,
    x_lim=None, y_lim=None, ax=None,
    do_quiver=False,
    plot_ref_vectorfield=False):
    
    if plot_initial_robot:
        
        robot.plot2D(ax=ax)
        
        ax.plot(robot.pose.position[0], robot.pose.position[1],
                "o", color='black', markersize=13, zorder=5)

        data_points = sample_environment.get_surface_points(
            center_position=robot.pose.position,
            )

        fast_avoider.update_laserscan(data_points)
        fast_avoider.update_reference_direction(in_robot_frame=False)

        # Store all
        initial_velocity = dynamical_system.evaluate(robot.pose.position)
        modulated_velocity = fast_avoider.avoid(initial_velocity)

        ax.plot(data_points[0, :], data_points[1, :], "o", color="k")

        arrow_scale = 0.5
        arrow_width = 0.07
        arrow_headwith = 0.4
        margin_velocity_plot = 1e-3

        # ax.arrow(
        #         robot.pose.position[0],
        #         robot.pose.position[1],
        #         arrow_scale * initial_velocity[0],
        #         arrow_scale * initial_velocity[1],
        #         width=arrow_width,
        #         head_width=arrow_headwith,
        #         # color="g",
        #         color="#008080",
        #         label="Initial velocity",
        #     )

        # ax.arrow(
        #     robot.pose.position[0],
        #     robot.pose.position[1],
        #     arrow_scale * modulated_velocity[0],
        #     arrow_scale * modulated_velocity[1],
        #     width=arrow_width,
        #     head_width=arrow_headwith,
        #     # color="b",
        #     # color='#213970',
        #     color="#000080",
        #     label="Modulated velocity",
        # )
        
        ax.arrow(
            robot.pose.position[0],
            robot.pose.position[1],
            fast_avoider.reference_direction[0],
            fast_avoider.reference_direction[1],
            color='#9b1503',
            width=arrow_width,
            head_width=arrow_headwith,
            label="Reference [summed]"
            )

        ax.arrow(
            robot.pose.position[0],
            robot.pose.position[1],
            fast_avoider.obstacle_avoider.reference_direction[0],
            fast_avoider.obstacle_avoider.reference_direction[1],
            color='#CD7F32',
            width=arrow_width,
            head_width=arrow_headwith,
            label="Reference [analytic]"
            )

        ax.arrow(
            robot.pose.position[0],
            robot.pose.position[1],
            fast_avoider.lidar_avoider.reference_direction[0],
            fast_avoider.lidar_avoider.reference_direction[1],
            color='#3d3635',
            width=arrow_width,
            head_width=arrow_headwith,
            label="Reference [sampled]"
            )

        ax.legend(loc="upper left", fontsize=12)

    
    nx = ny = n_resolution
    x_vals, y_vals = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
    )

    positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1)))

    velocities_init = np.zeros(positions.shape)
    velocities_mod = np.zeros(positions.shape)

    reference_dirs = np.zeros(positions.shape)
    norm_dirs = np.zeros(positions.shape)

    for it in range(positions.shape[1]):
        robot.pose.position = positions[:, it]

        is_inside_an_obstacle = False
        for obs in fast_avoider.obstacle_environment:
            if obs.get_gamma(robot.pose.position, in_global_frame=True) < 1:
                is_inside_an_obstacle = True
                break
            
        if is_inside_an_obstacle:
            continue

        if sample_environment.is_inside(
            position=robot.pose.position,
            margin=robot.control_radius):
            continue

        data_points = sample_environment.get_surface_points(
            center_position=robot.pose.position,
            )
        
        # _, _, relative_distances = robot.get_relative_positions_and_dists(
            # data_points, in_robot_frame=False
        # )

        # if any(relative_distances < 0):
            # continue

        fast_avoider.update_laserscan(data_points)

        fast_avoider.update_reference_direction(in_robot_frame=False)
        
        velocities_init[:, it] = dynamical_system.evaluate(robot.pose.position)
        velocities_mod[:, it] = fast_avoider.avoid(velocities_init[:, it])

        # Reference and normal dir
        reference_dirs[:, it] = fast_avoider.reference_direction
        norm_dirs[:, it] = fast_avoider.normal_direction


    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))
        
    # ax.plot(data_points[0, :], data_points[1, :], "k.")

    if do_quiver:
        ax.quiver(
            positions[0, :],
            positions[1, :],
            velocities_mod[0, :],
            velocities_mod[1, :],
            angles="xy",
            scale_units="xy",
            # scale=scale_vel,
            # width=arrow_width,
            color="blue",
    )

    else:
       ax.streamplot(
        positions[0, :].reshape(nx, ny),
        positions[1, :].reshape(nx, ny),
        velocities_mod[0, :].reshape(nx, ny),
        velocities_mod[1, :].reshape(nx, ny),
        # angles="xy",
        # scale_units="xy",
        # scale=scale_vel,
        # width=arrow_width,
        color="blue",
        zorder=-4
    )

    visualize_obstacles(sample_environment, ax=ax)
    plot_obstacles(ax=ax, obstacle_container=fast_avoider.obstacle_environment,
                       x_lim=x_lim, y_lim=y_lim, drawVelArrow=False)

    if hasattr(dynamical_system, 'attractor_position'):
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

    if plot_ref_vectorfield:
        # _, ax_ref = plt.subplots(1, 1, figsize=(10, 6))
        ax_ref = ax

        ax_ref.quiver(
            positions[0, :],
            positions[1, :],
            reference_dirs[0, :],
            reference_dirs[1, :],
            angles="xy",
            scale_units="xy",
            # scale=scale_vel,
            # width=arrow_width,
            color="red",
        )
        
        # visualize_obstacles(main_environment, ax=ax_ref)

        ax_ref.set_xlim(x_lim)
        ax_ref.set_ylim(y_lim)

        if not show_ticks:
            ax_ref.axes.xaxis.set_visible(False)
            ax_ref.axes.yaxis.set_visible(False)

        ax_ref.set_aspect("equal")

    return ax


def vectorfield_with_scenario_mixed(save_figure=False):
    # start_point = np.array([9, 6])
    start_point = np.array([13, 4])
    x_lim = [0, 18]
    y_lim = [0, 8]

    control_radius = 0.6

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([17, 3.0]), maximum_velocity=1.0
    )
    
    analytic_environment = GradientContainer()
    analytic_environment.append(
        CircularObstacle(center_position=np.array([14, 7]),
                orientation=-20*np.pi/180,
                radius=0.5,
                margin_absolut=control_radius,
                linear_velocity=np.array([-0.3, -0.5]),
                )
        )

    analytic_environment.append(
        CircularObstacle(center_position=np.array([6, 1]),
                         orientation=-20*np.pi/180,
                         radius=0.5,
                         margin_absolut=control_radius,
                         # linear_velocity=np.array([-0.5, 0.2]),
               )
        )

    sampled_environment = ShapelySamplingContainer(n_samples=50)
    ellipse = shapely.affinity.scale(shapely.geometry.Point(14, 1).buffer(1), 2, 1)
    ellipse = shapely.affinity.rotate(ellipse, 50)
    sampled_environment.add_obstacle(ellipse)
    
    sampled_environment.add_obstacle(shapely.geometry.box(5, 4, 7, 8))

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

    # Plot the vectorfield around the robot
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    static_visualization_of_sample_avoidance_mixed(
        robot=robot,
        n_resolution=100,
        dynamical_system=initial_dynamics,
        fast_avoider=mixed_avoider,
        plot_initial_robot=True,
        sample_environment=sampled_environment,
        # show_ticks=False,
        show_ticks=True,
        x_lim=x_lim, y_lim=y_lim,
        ax=ax,
        # do_quiver=True,
    )

    if save_figure:
        figure_name = "multiple_avoiding_obstacles_mixed"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


def animation_with_mixed(save_animation):
    start_point = np.array([3, 4])

    x_lim = [0, 18]
    y_lim = [0, 8]

    control_radius = 0.6

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([17, 3.0]), maximum_velocity=1.0
    )
    
    analytic_environment = GradientContainer()
    analytic_environment.append(
        CircularObstacle(center_position=np.array([14, 7]),
                orientation=-20*np.pi/180,
                radius=0.5,
                margin_absolut=control_radius,
                linear_velocity=np.array([-0.3, -0.5]),
                )
        )

    analytic_environment.append(
        CircularObstacle(center_position=np.array([3, 1]),
                         orientation=-20*np.pi/180,
                         radius=0.5,
                         margin_absolut=control_radius,
                         # linear_velocity=np.array([-0.5, 0.2]),
               )
        )

    # Sample Environment
    sampled_environment = ShapelySamplingContainer(n_samples=50)
    ellipse = shapely.affinity.scale(shapely.geometry.Point(14, 1).buffer(1), 2, 1)
    ellipse = shapely.affinity.rotate(ellipse, 50)
    sampled_environment.add_obstacle(ellipse)
    
    sampled_environment.add_obstacle(shapely.geometry.box(5, 4, 7, 8))

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

    plt.close('all')

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


if (__name__) == "__main__":
    plt.ion()
    plt.close('all')
    
    # vectorfield_with_scenario_mixed(save_figure=False)
    animation_with_mixed(save_animation=True)
