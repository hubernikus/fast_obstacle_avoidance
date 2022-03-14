""" Script to create plots. """
# Author: Lukas Huber
# Created: 2021-02-22
# Email: lukas.huber@epfl.ch

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

from vartools.animator import Animator
from vartools.states import ObjectPose
from vartools.dynamical_systems import LinearSystem

from dynamic_obstacle_avoidance.obstacles import EllipseWithAxes
from dynamic_obstacle_avoidance.obstacles import CuboidXd
from dynamic_obstacle_avoidance.containers import ObstacleContainer

# from dynamic_obstacle_avoidance.obstacles import Ellipse
from dynamic_obstacle_avoidance.visualization import plot_obstacles

from fast_obstacle_avoidance.control_robot import BaseRobot
from fast_obstacle_avoidance.obstacle_avoider import FastObstacleAvoider


class FourObstacleAnimator(Animator):
    def setup(self):
        start_point = np.array([9, 6])
        self.robot = BaseRobot(pose=ObjectPose(position=start_point, orientation=0))

        self.environment = ObstacleContainer()
        self.environment.append(
            EllipseWithAxes(
                center_position=np.array([2, 6]),
                orientation=0,
                axes_length=np.array([0.4, 0.8]),
                margin_absolut=self.robot.control_radius,
            )
        )

        self.environment.append(
            CuboidXd(
                center_position=np.array([5.5, 2]),
                orientation=-40 * np.pi / 180,
                axes_length=np.array([0.8, 0.8]),
                margin_absolut=self.robot.control_radius,
                angular_velocity=10 * np.pi / 180,
            )
        )

        self.environment.append(
            EllipseWithAxes(
                center_position=np.array([14, 7]),
                orientation=-20 * np.pi / 180,
                axes_length=np.array([0.3, 0.9]),
                margin_absolut=self.robot.control_radius,
                linear_velocity=np.array([-0.7, -0.2]),
            )
        )

        self.environment.append(
            EllipseWithAxes(
                center_position=np.array([13, 2]),
                orientation=30 * np.pi / 180,
                axes_length=np.array([0.3, 2.1]),
                margin_absolut=self.robot.control_radius,
            )
        )

        self.initial_dynamics = LinearSystem(
            attractor_position=np.array([17, 1.0]), maximum_velocity=1.0
        )

        self.avoider = FastObstacleAvoider(
            obstacle_environment=self.environment,
            robot=self.robot,
        )

        # Visualize Afterwards
        self.fig, self.ax = plt.subplots(1, 1, figsize=(16, 8))

    def update_step(self, ii):
        if not (ii % 10):
            print(f"It {ii}")

        # Initial Dynmics and Avoidance
        self.initial_velocity = self.initial_dynamics.evaluate(self.robot.pose.position)
        self.modulated_velocity = self.avoider.avoid(self.initial_velocity)

        self.robot.pose.position = (
            self.robot.pose.position + self.modulated_velocity * self.dt_simulation
        )

        self.plot_environment(ii=ii)

    def plot_environment(self, ii):
        self.ax.clear()

        plot_obstacles(
            obstacle_container=self.environment,
            ax=self.ax,
            x_lim=[0, 20],
            y_lim=[0, 10],
            draw_reference=True,
        )

        self.ax.plot(
            self.robot.pose.position[0],
            self.robot.pose.position[1],
            "o",
            color="k",
            markersize=18,
        )

        self.ax.plot(
            self.initial_dynamics.attractor_position[0],
            self.initial_dynamics.attractor_position[1],
            "k*",
            linewidth=18.0,
            markersize=18,
            zorder=5,
        )

    def has_converged(self, ii):
        """Return 0 if still going, and
        >0 : has converged at `ii`
        -1 : stuck somewher away from the attractor
        """
        if (
            LA.norm(self.robot.pose.position - self.initial_dynamics.attractor_position)
            < 1e-1
        ):
            # Check distance to attractor
            return ii

        elif LA.norm(self.modulated_velocity) < 1e-2:
            #  Check Velocity
            return 1

        return 0


def main(it_max=1000):
    plt.ion()

    my_animator = FourObstacleAnimator(
        it_max=400,
        dt_simulation=0.05,
    )
    my_animator.setup()
    my_animator.run()


if (__name__) == "__main__":
    main()
