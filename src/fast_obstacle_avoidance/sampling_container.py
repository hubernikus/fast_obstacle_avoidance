""" Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2021-12-14
# Email: lukas.huber@epfl.ch

import numpy as np
from numpy import linalg as LA

from abc import ABC, abstractmethod

import shapely

import matplotlib.pyplot as plt


def visualize_obstacles(container, ax=None, x_lim=None, y_lim=None):
    if ax is None:
        fig, ax = plt.subplots()

    for obs in container.environment:
        xx, yy = obs.exterior.xy
        ax.plot(xx, yy, color="black", alpha=0.3)

        if obs.is_boundary:
            if x_lim is None:
                x_min = min(xx)
                x_max = max(xx)
                delta_x = x_max - x_min

            if y_lim is None:
                y_min = min(yy)
                y_max = max(yy)
                delta_y = y_max - y_min

            enclosing_path = plt.Rectangle(
                (x_min - delta_x, y_min - delta_y),
                3 * delta_x,
                3 * delta_y,
                alpha=0.1,
                zorder=-5,
            )
            enclosing_path.set_color("black")
            ax.add_patch(enclosing_path)

            polygon_path = plt.Polygon(np.vstack((xx, yy)).T, alpha=1.0, zorder=-4)
            polygon_path.set_color("white")
            ax.add_patch(polygon_path)

        else:
            polygon_path = plt.Polygon(np.vstack((xx, yy)).T, alpha=0.1, zorder=-4)
            polygon_path.set_color("black")
            ax.add_patch(polygon_path)

    return ax


class SampledObstacle(ABC):
    """Sampled Obstacle Wrapper which allows for additional properties and
    custom construction."""

    def __init__(self, is_boundary=False):
        self.is_boundary = is_boundary

    @property
    def obstacle(self):
        return self._obstacle

    @obstacle.setter
    def obstacle(self, value):
        self._obstacle = value

    @property
    def exterior(self):
        return self._obstacle.exterior

    def intersection(self, shapely_line):
        return self._obstacle.intersection(shapely_line)

    def contains(self, shapely_point):
        return self._obstacle.contains(shapely_point)


class SampledEllipse(SampledObstacle):
    def __init__(
        self,
        position=None,
        axes_length=None,
        orientation_in_degree=0,
        obstacle=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        if obstacle is not None:
            position = obstacle.center_position
            axes_length = obstacle.axes_length

            if obstacle.orientation is None:
                orientation_in_degree = 0
            else:
                orientation_in_degree = obstacle.orientation * 180 / np.pi

        ellipse = shapely.geometry.Point(position[0], position[1]).buffer(1)
        ellipse = shapely.affinity.scale(
            ellipse, axes_length[0] * 0.5, axes_length[1] * 0.5
        )
        self._obstacle = shapely.affinity.rotate(ellipse, orientation_in_degree)


class SampledCuboid(SampledObstacle):
    def __init__(
        self,
        position=None,
        axes_length=None,
        orientation_in_degree=0,
        obstacle=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        if obstacle is not None:
            position = obstacle.center_position
            axes_length = np.array(obstacle.axes_length)
            if obstacle.orientation is None:
                orientation_in_degree = 0
            else:
                orientation_in_degree = obstacle.orientation * 180 / np.pi

        semiaxes = np.array(axes_length) * 0.5
        cuboid = shapely.geometry.box(
            position[0] - semiaxes[0],
            position[1] - semiaxes[1],
            position[0] + semiaxes[0],
            position[1] + semiaxes[1],
        )

        self._obstacle = shapely.affinity.rotate(cuboid, orientation_in_degree)


class SampledSphere(SampledObstacle):
    def __init__(self, position=None, radius=None, obstacle=None, **kwargs):
        super().__init__(**kwargs)

        if obstacle is not None:
            position = obstacle.center_position
            radius = obstacle.radius

        self._obstacle = shapely.geometry.Point(position[0], position[1]).buffer(radius)


class ShapelySamplingContainer:
    # Only implemented for two dimensional case
    dimension = 2

    def __init__(self, environment=None, n_samples=10):
        if environment is None:
            self.environment = []
        else:
            self.environment = environment

        self.n_samples = n_samples

    def get_center_position(self, ii):
        """Returns geometric center of all surface points."""
        # TODO: maybe checkout the 'kernel' property of shapelies
        xy_vals = self.environment[ii].exterior.xy
        return np.mean(xy_vals, axis=1)

    def is_inside(self, position, margin=0):
        """Checks if the position is inside any of the obstacles."""
        if not margin:
            point = shapely.geometry.Point(position)

        for ii, obs in enumerate(self.environment):
            if margin:
                # Move the point along the margin
                center_position = self.get_center_position(ii)
                rel_pos = center_position - position

                rel_pos_norm = LA.norm(rel_pos)
                if rel_pos_norm <= margin:
                    return True

                if obs.is_boundary:
                    temp_pos = position - rel_pos / LA.norm(rel_pos) * margin
                else:
                    temp_pos = position + rel_pos / LA.norm(rel_pos) * margin
                point = shapely.geometry.Point(temp_pos)

            if obs.contains(point) != obs.is_boundary:
                # Inside and not boundary
                # OR outside and and boundary
                return True

        return False

    def __len__(self):
        return len(self.environment)

    def add_obstacle(self, obstacle):
        self.environment.append(obstacle)

    def create_ellipse(self, **kwargs):
        self.environment.append(SampledEllipse(**kwargs))

    def create_sphere(self, **kwargs):
        self.environment.append(SampledSphere(**kwargs))

    def create_cuboid(self, **kwargs):
        self.environment.append(SampledCuboid(**kwargs))

    def get_surface_points(
        self, center_position, n_samples=None, null_direction=None, dist_max=1e3
    ):
        if not n_samples:
            n_samples = self.n_samples
        sample_points = np.zeros((self.dimension, n_samples))
        sample_dist = np.ones(n_samples) * (-1)

        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
        if null_direction is not None:
            angles = angles + np.arctan2(null_direction[1], null_direction[0])

        for ii in range(n_samples):
            direction = np.array([np.cos(angles[ii]), np.sin(angles[ii])])
            shapely_line = shapely.geometry.LineString(
                [center_position, center_position + direction * dist_max]
            )

            for obs in self.environment:
                intersection_line = list(obs.intersection(shapely_line).coords)

                if len(intersection_line):
                    # TODO: if sure, this could be done with only the closest...
                    # dists = LA.norm(np.array(intersection_line)
                    # - np.tile(center_position, (len(intersection_line), 1)),
                    # axis=1)
                    # min_dist = min(dists)
                    # breakpoint()
                    # sample_list[:, ii] = min(sample_list[:, ii], min_dist)
                    if obs.is_boundary:
                        surface_point = np.array(intersection_line[1])
                    else:
                        surface_point = np.array(intersection_line[0])

                    min_dist = LA.norm(surface_point - center_position)

                    if sample_dist[ii] < 0 or sample_dist[ii] > min_dist:
                        sample_points[:, ii] = surface_point
                        sample_dist[ii] = min_dist

        return sample_points[:, sample_dist > 0]
