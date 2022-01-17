""" Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2021-12-14
# Email: lukas.huber@epfl.ch

import sys
import os
from timeit import default_timer as timer

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

# import pandas as pd
import rosbag

from vartools.states import ObjectPose
from vartools.dynamical_systems import LinearSystem
from vartools.animator import Animator

from dynamic_obstacle_avoidance.visualization import plot_obstacles

from fast_obstacle_avoidance.control_robot import QoloRobot
from fast_obstacle_avoidance.utils import laserscan_to_numpy
from fast_obstacle_avoidance.obstacle_avoider import MixedEnvironmentAvoider

from fast_obstacle_avoidance.laserscan_utils import import_first_scans, reset_laserscan
from fast_obstacle_avoidance.laserscan_utils import import_first_scan_and_crowd


def run_vectorfield_mixed(qolo):
    """Draw the vectorfield mixed"""

    my_avoider = MixedEnvironmentAvoider(qolo)
    my_avoider.update_laserscan(qolo.get_allscan())

    my_avoider.update_reference_direction()

    # x_lim = [-3, 4]
    # y_lim = [-3, 3]
    limit_scan = my_avoider.get_scan_without_ocluded_points()

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(
        my_avoider.laserscan[0, :],
        my_avoider.laserscan[1, :],
        ".",
        # color=np.array([177, 124, 124]) / 255.0,
        color="black",
        alpha=0.2,
        zorder=-1,
    )

    ax.plot(
        limit_scan[0, :],
        limit_scan[1, :],
        ".",
        # color=np.array([177, 124, 124]) / 255.0,
        # color='b',
        color="black",
        zorder=-1,
    )

    # for obs in my_avoider.obstacle_avoider.obstacle_environment:
    # obs.margin_absolut += qolo.radius

    plot_obstacles(
        ax,
        my_avoider.obstacle_avoider.obstacle_environment,
        showLabel=False,
        draw_reference=False,
        velocity_arrow_factor=1.0,
        # noTicks=True,
        x_lim=[-7.5, 7.5],
        y_lim=[-7.5, 7.5],
    )

    ax.grid()

    # breakpoint()


if (__name__) == "__main__":
    plt.close("all")
    plt.ion()

    # run_vectorfield_mixed()
    do_the_import = True
    if not 'qolo' in vars() or not 'qolo' in globals() or do_the_import:
        qolo = QoloRobot(
            pose=ObjectPose(position=[0.7, -0.7], orientation=30 * np.pi / 180)
        )

        import_first_scan_and_crowd(
            robot=qolo,
            bag_name="2021-12-03-18-21-29.bag",
            bag_dir="/home/lukas/Code/data_qolo/outdoor_recording/",
            start_time=None,
        )

    run_vectorfield_mixed(qolo=qolo)
