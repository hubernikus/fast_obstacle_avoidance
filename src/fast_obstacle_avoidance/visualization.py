""" Script to create animations."""
# Author: Lukas Huber
# Created: 2021-02-23
# Email: lukas.huber@epfl.ch

from timeit import default_timer as timer

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import shapely

from vartools.animator import Animator
from vartools.states import ObjectPose
from vartools.dynamical_systems import ConstantValue
from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.control_robot import QoloRobot

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer
from fast_obstacle_avoidance.sampling_container import visualize_obstacles


class BaseFastAnimator(Animator):
    def setup(
        self,
        robot,
        initial_dynamics,
        avoider,
        environment,
        x_lim=[-10, 10],
        y_lim=[-10, 10],
        show_reference=False,
        show_ticks=False,
        show_velocity=True,
        convergence_distance=1e-2,
        convergence_velocity=1e-3,
    ):
        self.dimension = 2

        self.robot = robot

        self.initial_dynamics = initial_dynamics
        self.avoider = avoider
        self.environment = environment

        self.show_reference = show_reference
        self.show_ticks = show_ticks
        self.show_velocity = show_velocity

        self.x_lim = x_lim
        self.y_lim = y_lim

        self.positions = np.zeros((self.dimension, self.it_max))

        self.velocities_init = np.zeros((self.dimension, self.it_max))
        self.velocities_mod = np.zeros((self.dimension, self.it_max))

        # Create
        self.fig, self.ax = plt.subplots(figsize=(16, 10))

        self.velocity_command = np.zeros(self.dimension)

        # Margins
        self.convergence_velocity = convergence_velocity
        self.convergence_distance = convergence_distance

    def has_converged(self):
        """ Return values:
        0 : No convvergence, agent still rolling
        1 : Velocity very low. Probably stuck somewhere
        2 : Very close to the attractor! Great success!
        """
        if LA.norm(self.velocity_command) < self.convergence_velocity:
            return 1

        if (
            LA.norm(sef.robot.pose.position - self.initial_dynamics.attractor_position)
            < self.convergence_distance
        ):
            return 2

        return 0


class LaserscanAnimator(BaseFastAnimator):
    def update_step(self, ii):
        # Print very few steps
        if not (ii % 10):
            print(f"It {ii}")

        self.positions[:, ii] = self.robot.pose.position

        data_points = self.environment.get_surface_points(
            center_position=self.robot.pose.position,
            null_direction=self.velocity_command,
        )

        self.avoider.update_reference_direction(data_points, in_robot_frame=False)

        # Store all
        self.initial_velocity = self.initial_dynamics.evaluate(self.robot.pose.position)
        self.modulated_velocity = self.avoider.avoid(self.initial_velocity)

        # Update step
        self.robot.pose.position = (
            self.robot.pose.position + self.modulated_velocity * self.dt_simulation
        )

        if self.do_the_plotting:
            self.plot_environment()



    def plot_environment(self):
        """ Plot the environment """
        # Restart plotting
        self.ax.clear()

        self.ax.plot(data_points[0, :], data_points[1, :], "o", color="k")

        self.ax.plot(
            self.robot.pose.position[0], self.robot.pose.position[1], "o", color="b"
        )

        visualize_obstacles(self.environment, ax=self.ax)

        self.ax.plot(self.positions[0, :ii], self.positions[1, :ii], "--", color="b")

        self.ax.set_aspect("equal")
        # self.ax.grid(True)

        self.ax.plot(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

        self.robot.plot2D(ax=self.ax)
        self.ax.plot(
            self.robot.pose.position[0],
            self.robot.pose.position[1],
            "o",
            color="black",
            markersize=13,
            zorder=5,
        )

        margin_velocity_plot = 2e-1
        arrow_scale = 0.5
        arrow_width = 0.07
        arrow_headwith = 0.4
        margin_velocity_plot = 1e-3

        drawn_arrow = False
        if LA.norm(self.initial_velocity) > margin_velocity_plot:
            self.ax.arrow(
                self.robot.pose.position[0],
                self.robot.pose.position[1],
                arrow_scale * self.initial_velocity[0],
                arrow_scale * self.initial_velocity[1],
                width=arrow_width,
                head_width=arrow_headwith,
                # color="g",
                color="#008080",
                label="Initial velocity",
            )
            darwn_arrow = True

        if LA.norm(self.modulated_velocity) > margin_velocity_plot:
            self.ax.arrow(
                self.robot.pose.position[0],
                self.robot.pose.position[1],
                arrow_scale * self.modulated_velocity[0],
                arrow_scale * self.modulated_velocity[1],
                width=arrow_width,
                head_width=arrow_headwith,
                # color="b",
                # color='#213970',
                color="#000080",
                label="Modulated velocity",
            )
            drawn_arrow = True

        if self.show_reference:
            self.ax.arrow(
                self.robot.pose.position[0],
                self.robot.pose.position[1],
                self.avoider.reference_direction[0],
                self.avoider.reference_direction[1],
                color="#9b1503",
                width=arrow_width,
                head_width=arrow_headwith,
                label="Reference direction",
            )

            drawn_arrow = True

        if drawn_arrow:
            self.ax.legend(loc="upper left", fontsize=18)

        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        # if not show_ticks:
        self.ax.axes.xaxis.set_visible(False)
        self.ax.axes.yaxis.set_visible(False)


class FastObstacleAnimator(BaseFastAnimator):
    def update_step(self, ii):
        # Print very few steps
        if not (ii % 10):
            print(f"It {ii}")

        # Update obstacle position:
        for obs in self.environment:
            obs.do_velocity_step(delta_time=self.dt_simulation)

        self.positions[:, ii] = self.robot.pose.position
        self.avoider.update_reference_direction(position=self.robot.pose.position)

        # Store all
        self.initial_velocity = self.initial_dynamics.evaluate(self.robot.pose.position)
        self.modulated_velocity = self.avoider.avoid(self.initial_velocity)

        # Update step
        self.robot.pose.position = (
            self.robot.pose.position + self.modulated_velocity * self.dt_simulation
        )

        if self.do_the_plotting:
            self.plot_environment()


    def plot_environment(self):
        """ Plot the environment """
        # Restart plotting
        self.ax.clear()

        # self.ax.plot(data_points[0, :], data_points[1, :], "o", color="k")

        self.ax.plot(
            self.robot.pose.position[0], self.robot.pose.position[1], "o", color="b"
        )

        # visualize_obstacles(self.environment, ax=self.ax)
        plot_obstacles(
            obstacle_container=self.environment,
            ax=self.ax,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
        )

        self.ax.plot(self.positions[0, :ii], self.positions[1, :ii], "--", color="b")

        self.ax.set_aspect("equal")
        # self.ax.grid(True)

        self.ax.plot(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

        self.robot.plot2D(ax=self.ax)
        self.ax.plot(
            self.robot.pose.position[0],
            self.robot.pose.position[1],
            "o",
            color="black",
            markersize=13,
            zorder=5,
        )

        margin_velocity_plot = 2e-1
        arrow_scale = 0.5
        arrow_width = 0.07
        arrow_headwith = 0.4
        margin_velocity_plot = 1e-3

        drawn_arrow = False
        if LA.norm(self.initial_velocity) > margin_velocity_plot:
            self.ax.arrow(
                self.robot.pose.position[0],
                self.robot.pose.position[1],
                arrow_scale * self.initial_velocity[0],
                arrow_scale * self.initial_velocity[1],
                width=arrow_width,
                head_width=arrow_headwith,
                # color="g",
                color="#008080",
                label="Initial velocity",
            )
            darwn_arrow = True

        if LA.norm(self.modulated_velocity) > margin_velocity_plot:
            self.ax.arrow(
                self.robot.pose.position[0],
                self.robot.pose.position[1],
                arrow_scale * self.modulated_velocity[0],
                arrow_scale * self.modulated_velocity[1],
                width=arrow_width,
                head_width=arrow_headwith,
                # color="b",
                # color='#213970',
                color="#000080",
                label="Modulated velocity",
            )
            drawn_arrow = True

        if self.show_reference:
            self.ax.arrow(
                self.robot.pose.position[0],
                self.robot.pose.position[1],
                self.avoider.reference_direction[0],
                self.avoider.reference_direction[1],
                color="#9b1503",
                width=arrow_width,
                head_width=arrow_headwith,
                label="Reference direction",
            )

            drawn_arrow = True

        if drawn_arrow:
            self.ax.legend(loc="upper left", fontsize=18)

        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        # if not show_ticks:
        self.ax.axes.xaxis.set_visible(False)
        self.ax.axes.yaxis.set_visible(False)


class MixedObstacleAnimator(BaseFastAnimator):
    def update_step(self, ii, show_ticks=False):
        # Print very few steps
        if not (ii % 10):
            print(f"It {ii}")

        # Update obstacle position:
        for obs in self.robot.obstacle_environment:
            obs.do_velocity_step(delta_time=self.dt_simulation)

        data_points = self.environment.get_surface_points(
            center_position=self.robot.pose.position,
            null_direction=self.velocity_command,
        )

        self.positions[:, ii] = self.robot.pose.position

        self.avoider.update_laserscan(data_points)
        # self.avoider.update_reference_direction(in_robot_frame=False)

        self.avoider.update_reference_direction(in_robot_frame=False)

        # Store all
        self.initial_velocity = self.initial_dynamics.evaluate(self.robot.pose.position)
        self.modulated_velocity = self.avoider.avoid(self.initial_velocity)

        # Speed up simulation
        self.modulated_velocity = (
            self.modulated_velocity
            / LA.norm(self.modulated_velocity)
            * LA.norm(self.initial_velocity)
        )

        # Update step
        self.robot.pose.position = (
            self.robot.pose.position + self.modulated_velocity * self.dt_simulation
        )

        if self.do_the_plotting:
            self.plot_environment()


    def plot_environment(self):
        # Restart plotting
        self.ax.clear()

        self.ax.plot(data_points[0, :], data_points[1, :], "o", color="k")

        self.ax.plot(
            self.robot.pose.position[0], self.robot.pose.position[1], "o", color="b"
        )

        # visualize_obstacles(self.environment, ax=self.ax)
        visualize_obstacles(self.environment, ax=self.ax)
        plot_obstacles(
            obstacle_container=self.robot.obstacle_environment,
            ax=self.ax,
            x_lim=self.x_lim,
            y_lim=self.y_lim,
        )

        self.ax.plot(self.positions[0, :ii], self.positions[1, :ii], "--", color="b")

        self.ax.set_aspect("equal")

        self.ax.plot(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

        self.robot.plot2D(ax=self.ax)
        self.ax.plot(
            self.robot.pose.position[0],
            self.robot.pose.position[1],
            "o",
            color="black",
            markersize=13,
            zorder=5,
        )

        margin_velocity_plot = 2e-1
        arrow_scale = 0.5
        arrow_width = 0.07
        arrow_headwith = 0.4
        margin_velocity_plot = 1e-3

        drawn_arrow = False

        if self.show_reference:
            self.ax.arrow(
                self.robot.pose.position[0],
                self.robot.pose.position[1],
                self.avoider.reference_direction[0],
                self.avoider.reference_direction[1],
                color="#9b1503",
                width=arrow_width,
                head_width=arrow_headwith,
                label="Reference [summed]",
            )

            self.ax.arrow(
                self.robot.pose.position[0],
                self.robot.pose.position[1],
                self.avoider.obstacle_avoider.reference_direction[0],
                self.avoider.obstacle_avoider.reference_direction[1],
                color="#CD7F32",
                width=arrow_width,
                head_width=arrow_headwith,
                label="Reference [obstacle]",
            )

            self.ax.arrow(
                self.robot.pose.position[0],
                self.robot.pose.position[1],
                self.avoider.lidar_avoider.reference_direction[0],
                self.avoider.lidar_avoider.reference_direction[1],
                color="#3d3635",
                width=arrow_width,
                head_width=arrow_headwith,
                label="Reference [sampled]",
            )
            drawn_arrow = True

        if self.show_velocity:
            if LA.norm(self.initial_velocity) > margin_velocity_plot:
                self.ax.arrow(
                    self.robot.pose.position[0],
                    self.robot.pose.position[1],
                    arrow_scale * self.initial_velocity[0],
                    arrow_scale * self.initial_velocity[1],
                    width=arrow_width,
                    head_width=arrow_headwith,
                    # color="g",
                    color="#008080",
                    label="Initial velocity",
                )
                darwn_arrow = True

            if LA.norm(self.modulated_velocity) > margin_velocity_plot:
                self.ax.arrow(
                    self.robot.pose.position[0],
                    self.robot.pose.position[1],
                    arrow_scale * self.modulated_velocity[0],
                    arrow_scale * self.modulated_velocity[1],
                    width=arrow_width,
                    head_width=arrow_headwith,
                    # color="b",
                    # color='#213970',
                    color="#000080",
                    label="Modulated velocity",
                )
                drawn_arrow = True

        if drawn_arrow:
            self.ax.legend(loc="upper left", fontsize=18)

        self.ax.set_xlim(self.x_lim)
        self.ax.set_ylim(self.y_lim)

        if not self.show_ticks:
            self.ax.axes.xaxis.set_visible(False)
            self.ax.axes.yaxis.set_visible(False)
