""" Script to create plots. """
# Author: Lukas Huber
# Created: 2021-03-02
# Email: lukas.huber@epfl.ch

import copy

from timeit import default_timer as timer

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt


def animation_comparison(start_position):
    main_environment = ShapelySamplingContainer(n_samples=50)
    ellipse = shapely.affinity.scale(
        shapely.geometry.Point(0.5, -0.5).buffer(1), 2.0, 1.5
    )
    ellipse = shapely.affinity.rotate(ellipse, 70)
    main_environment.add_obstacle(ellipse)

    main_environment.add_obstacle(shapely.geometry.box(-6, -1, -5, 1.5))

    robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
    robot.control_point = [0, 0]
    robot.control_radius = control_radius

    fast_avoider = SampledAvoider(
            robot=robot,
            weight_max_norm=1e8,
            weight_factor=4,
            weight_power=2.0,
            )

    my_animator = LaserscanAnimator(
        it_max=400,
        dt_simulation=0.05,
        )
    
    my_animator.setup(
            robot=robot,
            initial_dynamics=initial_dynamics,
            avoider=fast_avoider,
            environment=main_environment,
            x_lim=x_lim,
            y_lim=y_lim,
            show_reference=True
            )

    my_animator.run(save_animation=False)

    return my_animator.convergence_state


def main_comparison():
    dimension = 2
    
    n_modes = 2
    n_repetitions = 3

    x_start = -7
    x_attractor = 7
    y_vals = [-4, 4]

    convergence_states = np.zeros((n_modes, n_repetitions))
    
    for ii in range(n_repetitions):
        pos_start = np.zeros(dimension)
        pos_start[0] = x_start
        pos_start[1] = np.random.rand(1)*(y_vals[1] - y_vals[0]) + y_vals[0]

        pos_attractor = np.zoers(dimension)
        pos_start[0] = x_attractor
        pos_start[1] = np.random.rand(1)*(y_vals[1] - y_vals[0]) + y_vals[0]
