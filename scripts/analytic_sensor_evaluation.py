"""
Test how the normal behaves when integrating along a flat-surface (2D)
"""
import numpy as np


def test_lambda_functions(pow_r=1.0, pow_e=1.0, fact_e=0.1):
    # mag_value = np.linspace(0.1, 1000, 200)
    mag_value = np.logspace(-2, 3, num=200)

    ln2 = np.log(2)
    lambda_e = np.exp(-1 * fact_e * mag_value**pow_e)
    lambda_r = np.exp(-1 * ln2 * (mag_value**pow_r - 1)) - 1

    # import matplotlib.pyplot as plt

    plt.ion()
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 2))

    plt.plot(mag_value, lambda_r, label="lambda_r")
    plt.plot(mag_value, lambda_e, label="lambda_e")

    ax.set_xscale("log")

    plt.legend()
    plt.grid()

    plt.show()


def simple_normal_weight_stretching():

    vec_prod = np.linspace(-1, 1, 300)
    ind_close = vec_prod < (-np.sqrt(2) / 2)

    stretch = np.ones(vec_prod.shape)
    # stretch[ind_close] = (1./2 + np.sqrt(2)/2) +  (1.0/2 - np.sqrt(2)/2) * np.cos(
    # - np.pi* (vec_prod[ind_close] + np.sqrt(2)/2)/ (1- np.sqrt(2)/2))
    # stretch[ind_close] = np.sqrt(2)*np.cos(-1*vec_prod[ind_close])
    stretch[ind_close] = np.sqrt(2) * (-1) * vec_prod[ind_close]
    plt.close("all")
    fig, ax = plt.subplots(figsize=(7, 2))

    plt.plot(vec_prod, stretch)
    plt.show()


class QoloRobot:
    def __init__(self, delta_angle=0.007000000216066837):
        self.delta_angle = delta_angle
        self.radius_robot = 0.35
        # self.radius_min_norm = 0.40
        self.radius_min_norm = 0.3505

        self.weight_power = 2

        # self.weight_max_norm = 1 / (self.radius_min_norm - self.radius_robot)
        # 9.26948535e
        # self.weight_max_norm = 12151.17294101414

        self.weight_max_norm = 6.99580150e04

    def evaluate_reference(
        self, dist_horizontal_rel, dist_horizontal=None, delta_angle0=0
    ):

        dist_horizontal = (1 + dist_horizontal_rel) * self.radius_robot

        angles = np.hstack(
            (
                np.flip(np.arange(delta_angle0, -np.pi / 2.0, (-1) * self.delta_angle)),
                np.arange(delta_angle0 + self.delta_angle, np.pi / 2, self.delta_angle),
            )
        )

        points = np.vstack(
            (
                np.tan(angles) * dist_horizontal,
                np.ones(angles.shape) * (-1 * dist_horizontal),
            )
        )

        dists = LA.norm(points, axis=0)
        dirs = (-1) * points / np.tile(dists, (2, 1))

        weights = 1 / (dists - self.radius_robot) ** self.weight_power

        sum_weights = np.sum(weights)

        if sum_weights > self.weight_max_norm:
            sum_weights = self.weight_max_norm

        if sum_weights > 1:
            weights = weights / sum_weights

        reference_dir = np.sum(dirs * np.tile(weights, (2, 1)), axis=1)

        print("ref dir", reference_dir)


def evaluation_reference_dir():
    my_qolo = QoloRobot()

    my_qolo.evaluate_reference(dist_horizontal_rel=0.1)


def evaluation_parameters():
    my_qolo = QoloRobot()
    n_points = 2 * np.pi / my_qolo.delta_angle
    print(f"Total number of points: {n_points}")

    delta_dist = 1 - np.cos(my_qolo.delta_angle / 2)
    print(f"Relative penetration distance of on plane {delta_dist}")
    print(
        f"Absolute penetration distance of on plane {delta_dist*my_qolo.radius_robot}"
    )

    # Horizontal detection at object-radius
    dist_horizontal = 2 * np.sin(my_qolo.delta_angle / 2) * my_qolo.radius_robot
    print(f"Horizontal distance on surface {dist_horizontal} m.")

    # Penetration of circular obstacle
    diameter_object = 0.01
    delta_dist = diameter_object - np.sqrt(diameter_object**2 - dist_horizontal**2)
    print(
        f"Penetration of an obstacle with diameter={diameter_object}m is {delta_dist}m."
    )
    print(
        f"Realtive is {delta_dist/diameter_object} with ~{diameter_object/dist_horizontal}"
        + f"control points."
    )

    # Triangluar obstacle
    d_penetration_max = 0.001
    min_angle = np.arctan((dist_horizontal / 2.0) / (d_penetration_max))
    print(f"Min angle for penetration <0.001 is {min_angle*180/np.pi} deg")


def plot_close_summation(num_angles=100, n_points=1000):
    delta_angle = 2 * np.pi / n_points
    rad_robot = 1

    # min_dist_on_surf = 10 * np.sin(delta_angle / 2) * rad_robot
    min_dist_on_surf = rad_robot * 0.1

    laserscan_angles = np.linspace(-np.pi / 2, np.pi / 2, n_points)
    dy_list = np.linspace(1e-4, 10, 1000)

    gap_width = min_dist_on_surf + rad_robot * 2

    r_unit_vec = np.vstack(((-1) * np.sin(delta_angle), np.cos(delta_angle)))

    ry_all = np.zeros(dy_list.shape)

    for ii, dy in enumerate(dy_list):
        dx = np.tan(laserscan_angles) * dy

        dists = np.zeros(dx.shape)
        ind_ingap = np.abs(dx) < gap_width / 2

        dists[ind_ingap] = np.abs(
            gap_width / 2 * 1 / np.sin(laserscan_angles[ind_ingap])
        )

        ind_outgap = np.logical_not(ind_ingap)
        dists[ind_outgap] = dy * 1 // np.cos(laserscan_angles[ind_outgap])

        weigths = 1 / dists

        import matplotlib.pyplot as plt

        r_vec = np.sum(r_unit_vec * np.tile(weigths, (2, 1)), axis=1)

        ry_all[ii] = r_vec[1]
        # breakpoint()

    plt.figure()
    # plt.plot(dx_list, r_vec)
    plt.plot(dy_list, ry_all)
    plt.ion()
    plt.show()
    breakpoint()


if (__name__) == "__main__":
    # evaluation_parameters()
    # evaluation_reference_dir()

    # test_lambda_functions()

    # simple_normal_weight_stretching()

    plot_close_summation()

    pass
