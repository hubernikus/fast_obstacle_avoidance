""" Script to create plots. """
# Author: Lukas Huber
# Created: 2021-02-22
# Email: lukas.huber@epfl.ch

from fast_obstacle_avoidance.obstacle_avoider._base import StretchingMatrixTrigonometric

import numpy as np
from numpy import linalg as LA


def test_trigonometric_eigenvalues():
    import matplotlib.pyplot as plt

    n_samples = 1000

    ref_norms = np.linspace(0, 5, n_samples)
    vals_tan = np.zeros(ref_norms.shape)
    vals_ref = np.zeros(ref_norms.shape)

    # ref and vel oposing
    ref_dir = np.array([1, 0])
    initial_vel = np.array([-1, 0])

    stretchin_matrix = StretchingMatrixTrigonometric()
    for ii, norm_val in enumerate(ref_norms):
        diag_matr = stretchin_matrix.get(
            norm_val, reference_direction=ref_dir, initial_velocity=initial_vel
        )

        vals_ref[ii] = diag_matr[0, 0]
        vals_tan[ii] = diag_matr[1, 1]

    plt.figure()

    plt.plot(ref_norms, vals_ref)
    plt.plot(ref_norms, vals_tan)


def test_various_surface_points():
    start_point = np.array([-2.5, 3])
    x_lim = [-4, 4]
    y_lim = [-1.5, 5.6]

    # dynamical_system = ConstantValue(velocity=[0, 1])
    initial_dynamics = LinearSystem(
        attractor_position=np.array([3.5, 1.3]), maximum_velocity=1.0
    )

    main_environment = ShapelySamplingContainer(n_samples=50)
    # main_environment.add_obstacle(shapely.geometry.box(-5, -1, 2, 1))

    # main_environment = ShapelySamplingContainer(n_samples=100)

    # main_environment.add_obstacle(shapely.geometry.box(-5, -1, 2, 1))
    # circle =   # type(circle)=polygon

    ellipse = shapely.affinity.scale(
        shapely.geometry.Point(0.5, -0.5).buffer(1), 2.0, 1.5
    )
    ellipse = shapely.affinity.rotate(ellipse, 90)

    main_environment.add_obstacle(ellipse)
    robot = QoloRobot(pose=ObjectPose(position=start_point, orientation=0))
    robot.control_radius = 0.6

    fast_avoider = SampledAvoider(
        robot=robot,
        evaluate_normal=False,
        # evaluate_normal=True,
        weight_max_norm=1e8,
        weight_factor=1,
        weight_power=2.0,
    )

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    static_visualization_of_sample_avoidance(
        robot=robot,
        dynamical_system=initial_dynamics,
        fast_avoider=fast_avoider,
        plot_initial_robot=True,
        main_environment=main_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
        do_quiver=True,
        plot_ref_vectorfield=True,
    )

    robot.pose.position = np.array([3.5, 1.3])
    explore_specific_point(
        robot=robot,
        dynamical_system=initial_dynamics,
        fast_avoider=fast_avoider,
        draw_robot=True,
        main_environment=main_environment,
        x_lim=x_lim,
        y_lim=y_lim,
        ax=ax,
    )


if (__name__) == "__main__":
    # test_trigonometric_eigenvalues()
    test_various_surface_points()
