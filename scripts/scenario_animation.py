""" Script to create animations."""
# Author: Lukas Huber
# Created: 2021-02-22
# Email: lukas.huber@epfl.ch

from timeit import default_timer as timer

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt

import shapely

from vartools.states import ObjectPose
from vartools.dynamical_systems import ConstantValue
from vartools.dynamical_systems import LinearSystem

from fast_obstacle_avoidance.obstacle_avoider import SampledAvoider
from fast_obstacle_avoidance.control_robot import QoloRobot
from vartools.animator import Animator

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer
from fast_obstacle_avoidance.sampling_container import visualize_obstacles
from fast_obstacle_avoidance.visualization import LaserscanAnimator


def single_polygon_animator():
    qolo = QoloRobot(pose=ObjectPose(position=[0.0, -3.4], orientation=0))

    # dynamical_system = ConstantValue(velocity=[0, 1])
    dynamical_system = LinearSystem(
        attractor_position=np.array([1, 3]), maximum_velocity=1.0
    )

    fast_avoider = SampledAvoider(
        robot=qolo,
        evaluate_normal=False,
        # evaluate_normal=True,
        weight_max_norm=1e4,
        weight_factor=2,
        weight_power=1.0,
    )

    main_environment = ShapelySamplingContainer(n_samples=100)

    main_environment.add_obstacle(shapely.geometry.box(-5, -1, 2, 1))

    my_animator = LaserscanAnimator(
        it_max=400,
        dt_simulation=0.1,
    )

    my_animator.setup(
        robot=qolo,
        initial_dynamics=dynamical_system,
        avoider=fast_avoider,
        environment=main_environment,
        x_lim=[-4, 4],
        y_lim=[-4, 4],
    )

    my_animator.run()


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    single_polygon_animator()
