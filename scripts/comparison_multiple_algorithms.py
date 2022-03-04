""" Script to create plots. """
# Author: Lukas Huber
# Created: 2021-03-02
# Email: lukas.huber@epfl.ch

import copy

from timeit import default_timer as timer

import numpy as np
from numpy import linalg as LA

import matplotlib.pyplot as plt


def animation_comparison():
    main_environment = ShapelySamplingContainer(n_samples=50)
    ellipse = shapely.affinity.scale(
        shapely.geometry.Point(0.5, -0.5).buffer(1), 2.0, 1.5
    )
    ellipse = shapely.affinity.rotate(ellipse, 70)
    main_environment.add_obstacle(ellipse)

    main_environment.add_obstacle(shapely.geometry.box(-6, -1, -5, 1.5))


class FastObstacleAnimator:
    pass
