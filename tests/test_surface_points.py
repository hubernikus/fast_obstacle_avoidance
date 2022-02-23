""" Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2022-02-23
# Email: lukas.huber@epfl.ch

import numpy as np

import shapely

from fast_obstacle_avoidance.sampling_container import ShapelySamplingContainer

from fast_obstacle_avoidance.visalization import explore_specific_point

def test_surface_points():
    main_environment = ShapelySamplingContainer()
    main_environment.add_obstacle(shapely.geometry.box(0, 2, -1, 1))

    ax = visualize_obstacles(main_environment)

    center_position = np.array(np.array([0, 3]))

    sample_list = main_environment.get_surface_points(
        center_position=center_position, n_samples=20
    )

    ax.plot(sample_list[0, :], sample_list[1, :], "o", color="k")
    ax.plot(center_position[0], center_position[1], "o", color="r")

    ax.set_aspect("equal")
    ax.grid(True)

    x_lim = [-2, 5]
    y_lim = [-1, 6]

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
