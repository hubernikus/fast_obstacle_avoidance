""" Script to create plots. """
# Author: Lukas Huber
# Created: 2021-02-22
# Email: lukas.huber@epfl.ch

import copy
from enum import Enum, auto

from timeit import default_timer as timer

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import shapely

from vartools.states import ObjectPose
from vartools.dynamical_systems import ConstantValue, LinearSystem
from vartools.dynamical_systems import plot_dynamical_system_streamplot
from vartools.dynamical_systems import plot_dynamical_system_quiver


from fast_obstacle_avoidance.obstacle_avoider import SampledClusterAvoider
from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.sampling_container import visualize_obstacles
from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer
from fast_obstacle_avoidance.sampling_container import SampledEllipse, SampledCuboid

from fast_obstacle_avoidance.visualization import LaserscanAnimator
from fast_obstacle_avoidance.visualization import (
    static_visualization_of_sample_avoidance,
)

# Comparison algorithm inspired on matlab
from fast_obstacle_avoidance.comparison.vfh_avoider import VFH_Avoider


class AlgorithmType(Enum):
    SAMPLED = 0
    MIXED = 1
    VFH = 2
    OBSTACLE = auto()
    MODULATED = auto()
    CLUSTERSAMPLED = auto()


def explore_specific_point(
    robot,
    dynamical_system,
    fast_avoider,
    main_environment,
    x_lim=None,
    y_lim=None,
    draw_robot=False,
    show_ticks=False,
    ax=None,
    draw_velocity=False,
):

    data_points = main_environment.get_surface_points(
        center_position=robot.pose.position,
    )

    eval_pos = robot.pose.position

    fast_avoider.debug_mode = True
    fast_avoider.update_reference_direction(data_points, in_robot_frame=False)

    initial_velocity = dynamical_system.evaluate(robot.pose.position)
    modulated_velocity = fast_avoider.avoid(initial_velocity)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 6))

    visualize_obstacles(main_environment, ax=ax)

    ax.plot(data_points[0, :], data_points[1, :], "o", color="black", markersize=11)
    ax.plot(
        robot.pose.position[0],
        robot.pose.position[1],
        "o",
        color="black",
        markersize=13,
    )

    if hasattr(fast_avoider, "clusterer"):
        ax.scatter(
            fast_avoider._cluster_centers[0, :],
            fast_avoider._cluster_centers[1, :],
            s=120,
            marker="o",
        )

    arrow_width = 0.04
    arrow_head_width = 0.2

    if hasattr(fast_avoider, "reference_direction"):
        reference_direction = fast_avoider.reference_direction.reshape(2, -1)
        for ii in range(reference_direction.shape[1]):
            ax.arrow(
                eval_pos[0],
                eval_pos[1],
                reference_direction[0, ii],
                reference_direction[1, ii],
                color="#9b1503",
                width=arrow_width,
                head_width=arrow_head_width,
            )

    if hasattr(fast_avoider, "normal_directions"):
        normal_directions = fast_avoider.normal_directions.reshape(2, -1)
        for ii in range(normal_directions.shape[1]):
            ax.arrow(
                eval_pos[0],
                eval_pos[1],
                normal_directions[0, ii],
                normal_directions[1, ii],
                color="#66cc66",
                width=arrow_width,
                head_width=arrow_head_width,
            )

    if hasattr(fast_avoider, "reference_directions"):
        reference_direction = fast_avoider.reference_directions.reshape(2, -1)
        for ii in range(reference_direction.shape[1]):
            ax.arrow(
                eval_pos[0],
                eval_pos[1],
                reference_direction[0, ii],
                reference_direction[1, ii],
                color="#9b1503",
                width=arrow_width,
                head_width=arrow_head_width,
            )

    if hasattr(fast_avoider, "ref_dirs"):
        ax.quiver(
            data_points[0, :],
            data_points[1, :],
            fast_avoider.ref_dirs[0, :],
            fast_avoider.ref_dirs[1, :],
            scale=10,
            color="#9b1503",
            # color="blue",
            # width=arrow_width,
            alpha=1.0,
            label="Reference direction",
        )

    if draw_velocity:
        arrow_scale = 0.5
        arrow_width = 0.07
        arrow_headwith = 0.4
        margin_velocity_plot = 1e-3

        ax.arrow(
            robot.pose.position[0],
            robot.pose.position[1],
            arrow_scale * initial_velocity[0],
            arrow_scale * initial_velocity[1],
            width=arrow_width,
            head_width=arrow_headwith,
            # color="g",
            color="#008080",
            label="Initial velocity",
        )

        ax.arrow(
            robot.pose.position[0],
            robot.pose.position[1],
            arrow_scale * modulated_velocity[0],
            arrow_scale * modulated_velocity[1],
            width=arrow_width,
            head_width=arrow_headwith,
            # color="b",
            # color='#213970',
            color="#000080",
            label="Modulated velocity",
        )

        ax.legend(loc="upper left", fontsize=12)

    plot_normal_direction = False
    if plot_normal_direction:
        self.ax.arrow(
            eval_pos[0],
            eval_pos[1],
            fast_avoider.normal_direction[0],
            fast_avoider.normal_direction[1],
            color="red",
            width=arrow_width,
            head_width=arrow_head_width,
        )

        self.ax.quiver(
            data_points[0, :],
            data_points[1, :],
            fast_avoider.normal_dirs[0, :],
            fast_avoider.normal_dirs[1, :],
            scale=10,
            color="red",
            # width=arrow_width,
            alpha=0.8,
        )

    robot.plot2D(ax=ax)

    ax.set_aspect("equal")

    if not show_ticks:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)

    # ax.grid(True)
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    return ax


def execute_avoidance_with_two_obstacles(
    save_figure=False,
    create_animation=False,
    algorithmtype: AlgorithmType = AlgorithmType.SAMPLED,
):
    """Visualizes
    (1) the repulsion from a specific point
    (2) the resulting vector-field."""

    # start_point = np.array([-2.5, 3])
    # start_point = np.array([-2.0, 4.2])
    # start_point = np.array([-2.0, 4.5])
    # start_point = np.array([3.0, 4.0])
    start_point = np.array([0.2, 2.5])
    x_lim = [-4, 4.5]
    y_lim = [-1.0, 5.6]

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([3.5, 1.3]), maximum_velocity=1.0
    )

    main_environment = ShapelySamplingContainer(n_samples=100)
    main_environment.add_obstacle(
        # SampledEllipse.from_obstacle(
        SampledCuboid.from_obstacle(
            position=np.array([0.5, 0.0]),
            orientation_in_degree=90,
            axes_length=np.array([4.0, 3.0]),
        )
    )

    main_environment.add_obstacle(
        # SampledEllipse.from_obstacle(
        SampledCuboid.from_obstacle(
            position=np.array([0.5, 5.0]),
            orientation_in_degree=90,
            axes_length=np.array([4.0, 3.0]),
        )
    )

    robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.4

    if algorithmtype == AlgorithmType.SAMPLED:
        fast_avoider = SampledAvoider(
            robot=robot,
            weight_max_norm=1e6,
            weight_factor=2 * np.pi / main_environment.n_samples * 10,
            weight_power=2.0,
        )
        fast_avoider.weight_factor = 2 * np.pi / main_environment.n_samples * 10
        fast_avoider.weight_power = 2.0
        algoname = "sampled"

    elif algorithmtype == AlgorithmType.CLUSTERSAMPLED:
        fast_avoider = SampledClusterAvoider(robot=robot)
        fast_avoider.weight_factor = 2 * np.pi / main_environment.n_samples * 1
        fast_avoider.weight_power = 0.5

        fast_avoider.weight_max_norm = 1e7
        algoname = "clustersampled"

    elif algorithmtype == AlgorithmType.VFH:
        fast_avoider = VFH_Avoider(
            robot=robot,
            # use_matlab=True,
            # matlab_engine=matlab_eng,
        )
        algoname = "vfh"
    else:
        warnings.warn(f"No matching algorithm found of type {algorithmtype}")
        return

    # Do the animation, only:
    if create_animation:
        simu_environment = copy.deepcopy(main_environment)
        # simu_environment.n_samples = 100

        simu_bot = copy.deepcopy(robot)
        simu_bot.pose.position = np.array([-2.5, 1])

        my_animator = LaserscanAnimator(
            it_max=400,
            dt_simulation=0.05,
            # dt_pause=0.1,
            animation_name="two_obstacle_avoidance" + "_" + algoname,
        )

        fast_avoider.robot = simu_bot

        my_animator.setup(
            robot=simu_bot,
            initial_dynamics=initial_dynamics,
            avoider=fast_avoider,
            environment=simu_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            plot_lidarlines=True,
            show_reference=True,
            show_lidarweight=False,
        )

        my_animator.run(save_animation=save_figure)

        print(f"Done the animation. Saved={save_figure}.")
        return

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    explore_specific_point(
        robot=robot,
        dynamical_system=initial_dynamics,
        fast_avoider=fast_avoider,
        main_environment=main_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        show_ticks=True,
        draw_velocity=True,
    )

    static_visualization_of_sample_avoidance(
        robot=robot,
        dynamical_system=initial_dynamics,
        fast_avoider=fast_avoider,
        plot_initial_robot=True,
        main_environment=main_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        plot_quiver=True,
        show_ticks=True,
        n_resolution=20,
    )

    if save_figure:
        figure_name = "visualize_avoiding_field"
        plt.savefig("figures/" + figure_name + ".png", bbox_inches="tight")


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")

    start_global_matlab_engine = False
    if start_global_matlab_engine and not "matlab_eng" in locals():
        import matlab
        import matlab.engine

        matlab_eng = matlab.engine.start_matlab()
        matlab_eng.addpath("src/fast_obstacle_avoidance/comparison/matlab")
        # str(Path("src") / "fast_obstacle_avoidance" / "comparison" / "matlab")

    execute_avoidance_with_two_obstacles(
        save_figure=False,
        create_animation=True,
        # algorithmtype=AlgorithmType.SAMPLED,
        algorithmtype=AlgorithmType.CLUSTERSAMPLED,
        # algorithmtype=AlgorithmType.VFH,
    )

    print("Done all.")
