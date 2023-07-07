""" Script to create plots. """
# Author: Lukas Huber
# Created: 2023-04-01
# Email: lukas.huber@epfl.ch

import numpy as np
import matplotlib.pyplot as plt

from vartools.states import ObjectPose
from vartools.animator import Animator
from vartools.dynamical_systems import LinearSystem

from fast_obstacle_avoidance.sampling_container import SampledCuboid
from fast_obstacle_avoidance.obstacle_avoider import SampledClusterAvoider
from fast_obstacle_avoidance.obstacle_avoider.sampled_cluster_avoider import (
    ClustererType,
)

from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.directional_learning import DirectionalKMeans
from fast_obstacle_avoidance.directional_learning import DirectionalSoftKMeans

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer

from fast_obstacle_avoidance.visualization import LaserscanAnimator


def get_multi_cluster_environment():
    main_environment = ShapelySamplingContainer(n_samples=100)
    main_environment.add_obstacle(
        # SampledEllipse.from_obstacle(
        SampledCuboid.from_obstacle(
            position=np.array([0.0, 1.0]),
            orientation_in_degree=0,
            axes_length=np.array([4.0, 1.0]),
        )
    )

    main_environment.add_obstacle(
        # SampledEllipse.from_obstacle(
        SampledCuboid.from_obstacle(
            position=np.array([1.5, 2.5]),
            orientation_in_degree=0,
            axes_length=np.array([1.0, 4.0]),
        )
    )

    # main_environment.add_obstacle(
    #     # SampledEllipse.from_obstacle(
    #     SampledCuboid.from_obstacle(
    #         position=np.array([0.0, 4.0]),
    #         orientation_in_degree=0,
    #         axes_length=np.array([4.0, 1.0]),
    #     )
    # )

    # main_environment.add_obstacle(
    #     # SampledEllipse.from_obstacle(
    #     SampledCuboid.from_obstacle(
    #         position=np.array([0, 6.0]),
    #         orientation_in_degree=0,
    #         axes_length=np.array([10.0, 1.0]),
    #     )
    # )
    return main_environment


class ClusterinAnimator(Animator):
    def setup(self):
        self.environment = get_multi_cluster_environment()

        start_point = np.array([-3.0, 1])
        self.robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
        self.robot.control_point = [0, 0]
        self.robot.control_radius = 0.4
        self.robot.pose.position = start_point

        # Create Initial Dynamics
        self.initial_dynamics = LinearSystem(
            attractor_position=np.array([3.5, 4.0]), maximum_velocity=1.0
        )

        # Setup avoider + parameters
        self.fast_avoider = SampledClusterAvoider(
            control_radius=self.robot.control_radius,
            clusterer=ClustererType.DBSCAN,
        )
        # self.fast_avoider.weight_factor = 2 * np.pi / self.environment.n_samples * 1
        # self.fast_avoider.weight_factor = 2 * np.pi / self.environment.n_samples * 10
        # self.fast_avoider.weight_power = 1.0 / 2
        # self.fast_avoider.weight_max_norm = 1e7

        # Initialize clusterer with first data points
        self.initial_velocity = self.initial_dynamics.evaluate(self.position)
        data_points = self.environment.get_surface_points(
            center_position=self.position,
            null_direction=self.initial_velocity,
        )

        # # clusterer = DirectionalKMeans()
        # clusterer = DirectionalSoftKMeans(stiffness=2.0)
        # clusterer.fit(data_points.T)
        # self.fast_avoider.clusterer = clusterer

        # Setup visualizer
        self.visualizer = LaserscanAnimator(it_max=self.it_max)
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
            figsize=(8, 6),
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
        if not ii % 10:
            print(f"Step {ii}")

        # Retrieve data-points from sampler (cartesian representation)
        data_points = self.environment.get_surface_points(
            center_position=self.position,
            null_direction=self.initial_velocity,
        )

        # Update the avoider
        self.fast_avoider.update_laserscan(data_points, in_robot_frame=False)

        # Modulate initial velocity
        self.initial_velocity = self.initial_dynamics.evaluate(self.position)
        self.modulated_velocity = self.fast_avoider.avoid(
            self.initial_velocity, self.position
        )

        # Time step
        self.position = self.position + self.modulated_velocity * self.dt_simulation

        self.visualize(ii=ii)

        # Label clusters
        cluster_centers = self.fast_avoider.get_cluster_centers()
        n_clusters = cluster_centers.shape[1]
        labels = self.fast_avoider.clusterer.labels_
        colors = ["red", "green", "blue", "orange", "yellow"]

        for ii in range(n_clusters):
            color = colors[ii]
            cluster_points = data_points[:, labels == ii]

            self.ax.scatter(
                cluster_points[0, :],
                cluster_points[1, :],
                100,
                color=color,
                alpha=0.5,
                marker=".",
                # markeredgewidth=2.0,
                zorder=2,
            )

            self.ax.scatter(
                cluster_centers[0, ii],
                cluster_centers[1, ii],
                200,
                color=color,
                marker="*",
                # markeredgewidth=2.0,
                zorder=2,
            )

            arrow_width = 0.05
            self.ax.arrow(
                cluster_centers[0, ii],
                cluster_centers[1, ii],
                self.fast_avoider.normal_directions[0, ii],
                self.fast_avoider.normal_directions[1, ii],
                color=color,
                width=arrow_width,
            )

            self.ax.arrow(
                self.position[0],
                self.position[1],
                # self.fast_avoider.reference_directions[0, ii],
                # self.fast_avoider.reference_directions[1, ii],
                self.fast_avoider.modulated_velocities[0, ii],
                self.fast_avoider.modulated_velocities[1, ii],
                color=color,
                width=arrow_width,
                label=f"weight={self.fast_avoider.normalized_weights[ii]:0.3f}",
            )

            self.ax.legend()


if (__name__) == "__main__":
    plt.ion()
    plt.close("all")

    my_animator = ClusterinAnimator(
        it_max=1000,
        dt_simulation=0.10,
        file_type=".gif",
        animation_name="two_obstacle_avoidance_clustersampled",
    )
    my_animator.setup()
    my_animator.run(save_animation=False)

    print("Done all.")
