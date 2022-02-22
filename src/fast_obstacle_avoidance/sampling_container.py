""" Script to evaluate the rosbag. """
# Author: Lukas Huber
# Created: 2021-12-14
# Email: lukas.huber@epfl.ch

import numpy as np
from numpy import linalg as LA

import shapely

import matplotlib.pyplot as plt

def visualize_obstacles(container, ax=None):
    if ax is None:
        fig, ax = plt.subplots()

    for obs in container.environment:
        xx, yy = obs.exterior.xy
        ax.plot(xx, yy, color='black', alpha=0.4)

        polygon_path = plt.Polygon(np.vstack((xx, yy)).T, alpha=0.2, zorder=-4)
        polygon_path.set_color('black')
        ax.add_patch(polygon_path)

    return ax

class ShapelySamplingContainer:
    # Only implemented for two dimensional case
    dimension = 2
    
    def __init__(self, environment=None, n_samples=10):
        if environment is None:
            self.environment = []
        else:
            self.environment = environment

        self.n_samples = n_samples

    def is_inside(self, position):
        """ Checks if the position is inside any of the obstacles."""
        point = shapely.geometry.Point(position)
        
        for obs in self.environment:
            if obs.contains(point):
                return True
        
        return False

    def add_obstacle(self, obstacle):
        self.environment.append(obstacle)
        
    def get_surface_points(self, center_position, n_samples=None, null_direction=None, dist_max=1e3):
        if not n_samples:
            n_samples = self.n_samples
        sample_points = np.zeros((self.dimension, n_samples)) 
        sample_dist  = np.ones(n_samples) * (-1)

        angles = np.linspace(0, 2*np.pi, n_samples, endpoint=False)
        if null_direction is not None:
            angles = angles + np.arctan2(null_direction[1], null_direction[0])
    
        for ii in range(n_samples):
            direction = np.array([np.cos(angles[ii]), np.sin(angles[ii])])
            shapely_line = shapely.geometry.LineString(
                [center_position, center_position+direction*dist_max]
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
                    min_dist = LA.norm(np.array(intersection_line[0]) - center_position)

                    if sample_dist[ii] < 0 or sample_dist[ii] > min_dist:
                        sample_points[:, ii] = intersection_line[0]
                        sample_dist[ii] = min_dist

        
        return sample_points[:, sample_dist > 0]
