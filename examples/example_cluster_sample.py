""" Script to create plots. """
# Author: Lukas Huber
# Created: 2023-04-01
# Email: lukas.huber@epfl.ch

import copy

from timeit import default_timer as timer

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import shapely

from vartools.states import ObjectPose
from vartools.animator import Animator
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


def get_two_obstacle_sampler():
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
    return main_environment


def execute_avoidance_with_two_obstacles(
    save_figure=False,
    create_animation=False,
    algorithmtype: AlgorithmType = AlgorithmType.SAMPLED,
    it_max: int = 170,
):
    """Visualizes
    (1) the repulsion from a specific point
    (2) the resulting vector-field."""

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([3.5, 1.3]), maximum_velocity=1.0
    )

    environment_sampler = get_two_obstacle_sampler()

    start_point = np.array([-2.5, 1])
    robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = 0.4
    # robot.pose.position = start_point

    fast_avoider = SampledClusterAvoider(robot=robot)
    fast_avoider.weight_factor = 2 * np.pi / environment_sampler.n_samples * 1
    fast_avoider.weight_power = 1.0 / 2
    fast_avoider.weight_max_norm = 1e7

    my_animator = LaserscanAnimator(
        it_max=it_max,
        dt_simulation=0.05,
        # dt_pause=0.1,
        file_type=".gif",
        animation_name="two_obstacle_avoidance" + "_" + "clustersampled",
    )

    my_animator.setup(
        robot=robot,
        initial_dynamics=initial_dynamics,
        avoider=fast_avoider,
        environment=environment_sampler,
        x_lim=[-4, 4.5],
        y_lim=[-1.0, 5.6],
        plot_lidarlines=True,
        show_reference=True,
        show_lidarweight=False,
    )

    my_animator.run(save_animation=save_figure)

    print(f"Done the animation. Saved={save_figure}.")


class GapPassingAnimator(Animator):
    def setup(self, start_point=np.array([-2.5, 1])):
        self.environment = get_two_obstacle_sampler()

        self.robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
        self.robot.control_point = [0, 0]
        self.robot.control_radius = 0.4
        self.robot.pose.position = np.array([-2.5, 1])

        # Create Initial Dynamics
        self.initial_dynamics = LinearSystem(
            attractor_position=np.array([3.5, 1.3]), maximum_velocity=1.0
        )

        # Setup avoider + parameters
        self.fast_avoider = SampledClusterAvoider(
            control_radius=self.robot.control_radius
        )
        self.fast_avoider.weight_factor = 2 * np.pi / self.environment.n_samples * 1
        self.fast_avoider.weight_power = 1.0 / 2
        self.fast_avoider.weight_max_norm = 1e7

        # Setup visualizer
        self.visualizer = LaserscanAnimator()
        self.visualizer.setup(
            robot=self.robot,
            initial_dynamics=self.initial_dynamics,
            avoider=self.fast_avoider,
            environment=self.environment,
            x_lim=[-4, 4.5],
            y_lim=[-1.0, 5.6],
            plot_lidarlines=True,
            show_reference=True,
            show_lidarweight=False,
        )

    @property
    def fig(self):
        return self.visualizer.fig

    @property
    def ax(self):
        return self.visualizer.ax

    @property
    def position(self):
        return self.robot.pose.position

    @position.setter
    def position(self, value):
        self.robot.pose.position = value

    def visualize(self, ii: int):
        self.ax.clear()

        # Update visualizer
        self.visualizer.ii = ii
        self.visualizer.positions[:, ii + 1] = self.robot.pose.position
        self.visualizer.initial_velocity = self.initial_velocity
        self.visualizer.modulated_velocity = self.modulated_velocity

        self.visualizer._plot_specific(ii=ii)
        self.visualizer._plot_sampled_environment(ii=ii)
        self.visualizer._plot_general(ii=ii)

    def update_step(self, ii: int):
        self.initial_velocity = self.initial_dynamics.evaluate(self.position)

        # Retrieve data-points from sampler (cartesian representation)
        data_points = self.environment.get_surface_points(
            center_position=self.position,
            null_direction=self.initial_velocity,
        )

        # Update the avoider
        self.fast_avoider.update_laserscan(data_points, in_robot_frame=False)

        # Modulate initial velocity
        self.modulated_velocity = self.fast_avoider.avoid(
            self.initial_velocity, self.position
        )

        # Time step
        self.position = self.position + self.modulated_velocity * self.dt_simulation

        self.visualize(ii=ii)


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")
    # execute_avoidance_with_two_obstacles(
    #     save_figure=False,
    #     create_animation=True,
    # )

    my_animator = GapPassingAnimator()
    my_animator.setup()
    my_animator.run()

    print("Done all.")
