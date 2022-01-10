"""
Test how the normal behaves when integrating along a flat-surface (2D)
"""
import numpy as np


class QoloRobot:
    def __init__(self, delta_angle=0.007000000216066837):
        self.delta_angle = delta_angle
        self.radius_robot = 0.35
        # self.radius_min_norm = 0.40
        self.radius_min_norm = 0.3505

        self.weight_power = 2

        self.weight_max_norm = 1 / (self.radius_min_norm - self.radius_robot)
        # 9.26948535e
        self.weight_max_norm = 12151.17294101414

    def evaluate_reference(self, dist_horizontal, delta_angle0=0):
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
            print("max norm", self.weight_max_norm)
            sum_weights = self.weight_max_norm

        if sum_weights > 1:
            weights = weights / sum_weights

        reference_dir = np.sum(dirs * np.tile(weights, (2, 1)), axis=1)

        print("ref dir", reference_dir)


def evaluation_reference_dir():
    my_qolo = QoloRobot()

    my_qolo.evaluate_reference(dist_horizontal=0.45)


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
    delta_dist = diameter_object - np.sqrt(diameter_object ** 2 - dist_horizontal ** 2)
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


if (__name__) == "__main__":
    # evaluation_parameters()
    evaluation_reference_dir()
    pass
