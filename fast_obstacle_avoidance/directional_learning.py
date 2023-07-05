import copy
import warnings
from typing import Optional
from enum import Enum, auto

import numpy as np

from sklearn.cluster import KMeans

from vartools.linalg import get_orthogonal_basis

# from vartools.linalg import get_orthogonal_basis as _get_orthonormal_basis

# def get_orthonormal_basis_row(vector):
#     """Returns orthonormal baiss with [vector ; orthonormal 1 ; orthornomal2 ; ..]"""
#     return _get_orthonormal_basis(vector)


def map_cartesian_to_infite_stereographic(
    basis: np.ndarray,
    directions: np.ndarray,
    max_value: float = 1e99,
    tangential_multiple: float = 2.0,
    cos_margin: float = 1e-6,
) -> np.ndarray:
    radiuses = np.linalg.norm(directions, axis=1)
    if not all(radiuses):
        breakpoint()

    normalized_directions = directions / np.tile(radiuses, (directions.shape[1], 1)).T

    # Make sure to catch numerical error of cosinus calculation

    stereograhic_value = normalized_directions @ basis

    # Cosinus of the directions
    cos_directions = stereograhic_value[:, 0]

    ind_parallel = cos_directions >= (1.0 - cos_margin)
    ind_opposing = cos_directions <= (-1.0 + cos_margin)

    # Apply general case
    ind_non_opposing = np.logical_not(np.logical_or(ind_parallel, ind_opposing))
    cos_values = cos_directions[ind_non_opposing]

    stereographic_factor = (
        tangential_multiple / (1 + cos_values) - tangential_multiple / 2.0
    )
    sin_values = np.sqrt(1.0 - cos_values * cos_values)
    stereograhic_value[ind_non_opposing, 1:] = (
        stereograhic_value[ind_non_opposing, 1:]
        * np.tile(
            stereographic_factor / sin_values, (stereograhic_value.shape[1] - 1, 1)
        ).T
    )

    # Treat the exceptions (infinitely far way)
    stereograhic_value[ind_opposing, 1:] = max_value

    stereograhic_value[:, 0] = radiuses

    if np.any(np.isnan(stereograhic_value)):
        breakpoint()
        raise Exception("Direction-space is 'nan'.")

    return stereograhic_value


def map_infinite_stereographic_to_cartesian(
    basis: np.ndarray,
    positions: np.ndarray,
    max_value: float = 1e99,
    tangential_multiple: float = 2.0,
    cos_margin: float = 1e-6,
):
    radii = positions[:, 0]
    stereographic_stretch = np.linalg.norm(positions[:, 1:], axis=1)

    ind_nonzero = stereographic_stretch > cos_margin
    # n_nonzeros = np.sum()
    # Inverse function is computed as follows
    # f = t / (1 + cos) - t / 2.0 => cos = 1 / (f/t + 1/2.0) - 1
    cos_values = 1 / (stereographic_stretch / tangential_multiple + 1 / 2.0) - 1.0
    sin_values = np.sqrt(1 - cos_values[ind_nonzero] * cos_values[ind_nonzero])

    directions = np.zeros_like(positions)
    factor = sin_values / stereographic_stretch[ind_nonzero] * radii[ind_nonzero]
    directions[ind_nonzero, 1:] = (
        positions[ind_nonzero, 1:] * np.tile(factor, (positions.shape[1] - 1, 1)).T
    )

    directions[:, 0] = cos_values * radii
    return directions @ basis.T


class DistanceWeightType(Enum):
    QUADRATIC = auto()
    EXPONENTIAL = auto()


def get_inverse_square_distance_weight(distances: np.ndarray) -> np.ndarray:
    weights = 1 / distances**2
    weights = weights / np.sum(weights)
    # return weights
    raise ValueError("This will lead to exploding values (!)")


def get_exponential_stiffness_weight(
    distances: np.ndarray, beta: float = 1.0
) -> np.ndarray:
    weights = np.exp(-beta * distances)
    weights = weights / np.sum(weights)
    return weights


class DirectionalKMeans:
    def __init__(
        self,
        max_iter: int = 100,
        conv_tol: float = 1e-4,
        n_clusters: int = 4,
    ) -> None:
        self.max_iter = max_iter
        self.conv_tol = conv_tol

        # TODO: automatically update if too big / too small
        self.n_clusters = n_clusters
        # self.cluster_centers_: np.ndarray

        self.n_samples_ = 0
        self.n_features_ = 0

    def initialize_centers(
        self, XX: np.ndarray, base_direction: np.ndarray, variance: float = 1.0
    ) -> np.ndarray:
        """Initialization is for no uniform -> more options can be set.

        For not- only random options exists"""
        min_radius = np.min(XX[:, 0])
        max_radius = np.max(XX[:, 0])
        radial = (
            np.random.rand(self.n_clusters, 1) * (max_radius - min_radius) + min_radius
        )
        tangential = np.random.randn(self.n_clusters, self.n_features_ - 1) * variance

        mapped_centers = np.hstack((radial, tangential))

        base = get_orthogonal_basis(base_direction)
        real_centers = map_infinite_stereographic_to_cartesian(
            basis=base, positions=mapped_centers
        )
        return real_centers

    def get_stereographic_center(self, kk: int) -> np.ndarray:
        """The stereographic center is always lying on the y = 0 axis, as it
        only has the radius."""
        center = np.zeros(self.n_features_)
        center[0] = np.linalg.norm(self.cluster_centers_[kk, :])
        return center

    def initialize_from_points(self, XX):
        ind_clusters = np.random.randint(
            low=0, high=XX.shape[0], size=(self.n_clusters)
        )
        return XX[ind_clusters, :]

    def fit(
        self,
        XX: np.ndarray,
        # base_direction: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        do_initialization: bool = False,
    ) -> None:
        self.n_samples_ = XX.shape[0]
        self.n_features_ = XX.shape[1]

        # self.cluster_centers_ = self.initialize_centers(XX, base_direction)
        if do_initialization or not hasattr(self, "cluster_centers_"):
            self.cluster_centers_ = self.initialize_from_points(XX)

        old_error = self.update_step_hard_boundary(XX, sample_weights=sample_weights)
        for ii in range(self.max_iter - 1):
            error = self.update_step_hard_boundary(XX, sample_weights=sample_weights)

            if abs(error - old_error) < self.conv_tol:
                print(f"Converged at {ii}")
                break

            old_error = error

        else:
            print("Ending without convergence.")

    def update_step_hard_boundary(
        self, XX: np.ndarray, sample_weights: Optional[np.ndarray] = None
    ) -> float:
        """Returns proximity-metric (mean-weighted-distances)"""
        transformed_positions = np.zeros(
            (self.n_samples_, self.n_features_, self.n_clusters)
        )
        distances = np.zeros((self.n_samples_, self.n_clusters))
        bases = np.zeros((self.n_features_, self.n_features_, self.n_clusters))

        for kk in range(self.n_clusters):
            bases[:, :, kk] = get_orthogonal_basis(self.cluster_centers_[kk, :])

            transformed_positions[:, :, kk] = map_cartesian_to_infite_stereographic(
                bases[:, :, kk], XX
            )

            distances[:, kk] = np.linalg.norm(
                transformed_positions[:, :, kk]
                - np.tile(self.get_stereographic_center(kk), (self.n_samples_, 1)),
                axis=1,
            )

        self.labels_ = np.argmin(distances, axis=1)
        it_empty = 0
        mean_squared_distance = 0

        for kk in range(self.n_clusters):
            ind_label = self.labels_ == kk

            if not np.sum(ind_label):
                warnings.warn("Assigning empty cluster to far away point.")
                if not it_empty:
                    min_dists = np.min(distances, axis=1)
                    furthest_dists = np.argsort(min_dists)

                it_empty += 1
                self.cluster_centers_[kk, :] = XX[-it_empty]
                continue

            if sample_weights is None:
                mapped_center = np.mean(transformed_positions[ind_label, :, kk], axis=0)
                mean_squared_distance = mean_squared_distance + np.sum(
                    distances[ind_label] * distances[ind_label]
                )

            else:
                mapped_center = np.mean(
                    transformed_positions[ind_label, :, kk]
                    * np.tile(sample_weights[ind_label], (self.n_features_, 1)).T,
                    axis=0,
                )
                weighted_dist = distances[ind_label] * sample_weights[ind_label]
                mean_squared_distance = mean_squared_distance + np.sum(
                    weighted_dist * weighted_dist
                )

            self.cluster_centers_[kk, :] = map_infinite_stereographic_to_cartesian(
                bases[:, :, kk], mapped_center.reshape(1, self.n_features_)
            )

            if np.any(np.isnan(self.cluster_centers_)):
                breakpoint()

        return mean_squared_distance / self.n_samples_

    def predict(self, XX: np.ndarray) -> np.ndarray[int]:
        if XX.shape[1] != self.n_features_:
            raise ValueError(f"Wrong data-dimension of {XX.shape}")

        transformed_positions = np.zeros(
            (XX.shape[0], self.n_features_, self.n_clusters)
        )
        distances = np.zeros((XX.shape[0], self.n_clusters))
        bases = np.zeros((self.n_features_, self.n_features_, self.n_clusters))

        for kk in range(self.n_clusters):
            bases[:, :, kk] = get_orthogonal_basis(self.cluster_centers_[kk, :])

            transformed_positions[:, :, kk] = map_cartesian_to_infite_stereographic(
                bases[:, :, kk], XX
            )

            distances[:, kk] = np.linalg.norm(
                transformed_positions[:, :, kk]
                - np.tile(self.get_stereographic_center(kk), (XX.shape[0], 1)),
                axis=1,
            )

        self.labels_ = np.argmin(distances, axis=1)

        return self.labels_


class DirectionalSoftKMeans:
    def __init__(
        self,
        max_iter: int = 100,
        conv_tol: float = 1e-4,
        n_clusters: int = 4,
        stiffness: float = 1.0,
        distance_type: DistanceWeightType = DistanceWeightType.EXPONENTIAL,
    ) -> None:
        self.max_iter = max_iter
        self.conv_tol = conv_tol

        self.distance_type = distance_type

        # TODO: automatically update if too big / too small
        self.n_clusters = n_clusters
        # self.cluster_centers_: np.ndarray

        self.n_samples_: int = 0
        self.n_features_: int = 0
        self.labels_: np.ndarray = np.array((0, 0))

        self.stiffness = stiffness

        # Counter to introduce a new cluster every 10 iterationsg
        self._it_fit = 0
        self.new_cluster_frequency = 10

    def get_stereographic_center(self, kk: int) -> np.ndarray:
        """The stereographic center is always lying on the y = 0 axis, as it
        only has the radius."""
        center = np.zeros(self.n_features_)
        center[0] = np.linalg.norm(self.cluster_centers_[kk, :])
        return center

    def initialize_from_points(self, XX):
        ind_clusters = np.random.randint(
            low=0, high=XX.shape[0], size=(self.n_clusters)
        )
        return XX[ind_clusters, :]

    def fit(
        self,
        XX: np.ndarray,
        sample_weights: Optional[np.ndarray] = None,
        do_initialization: bool = False,
    ) -> None:
        self.n_samples_ = XX.shape[0]
        self.n_features_ = XX.shape[1]

        # self.cluster_centers_ = self.initialize_centers(XX, base_direction)
        if do_initialization or not hasattr(self, "cluster_centers_"):
            self.cluster_centers_ = self.initialize_from_points(XX)

        self._it_fit += 1
        if not self._it_fit % self.new_cluster_frequency:
            print("Adding a new cluster.")
            self.add_cluster(XX)

        old_error = self.update_step_soft_boundary(XX, sample_weights=sample_weights)
        for ii in range(self.max_iter - 1):
            error = self.update_step_soft_boundary(XX, sample_weights=sample_weights)

            if abs(error - old_error) < self.conv_tol:
                print(f"Converged at {ii}")
                return

            old_error = error

        print("Ended without convergence.")

    def add_cluster(self, XX: np.ndarray) -> None:
        # TODO: maybe this could be integrated in the update_step to reduce computation
        distances = self.get_cluster_distances(XX)
        closest_distance = np.min(distances, axis=1)
        ind_furthest = np.argmax(closest_distance)

        self.cluster_centers_ = np.append(
            self.cluster_centers_, XX[ind_furthest, :].reshape(1, -1), axis=0
        )
        self.labels_[ind_furthest] = self.n_clusters
        self.n_clusters += 1

    def get_cluster_distances(self, XX) -> np.ndarray:
        distances = np.zeros((self.n_samples_, self.n_clusters))
        transformed_positions = np.zeros(
            (self.n_samples_, self.n_features_, self.n_clusters)
        )
        bases = np.zeros((self.n_features_, self.n_features_, self.n_clusters))
        for kk in range(self.n_clusters):
            bases[:, :, kk] = get_orthogonal_basis(self.cluster_centers_[kk, :])
            transformed_positions[:, :, kk] = map_cartesian_to_infite_stereographic(
                bases[:, :, kk], XX
            )
            distances[:, kk] = np.linalg.norm(
                transformed_positions[:, :, kk]
                - np.tile(self.get_stereographic_center(kk), (self.n_samples_, 1)),
                axis=1,
            )
        return distances

    def update_step_soft_boundary(
        self, XX: np.ndarray, sample_weights: Optional[np.ndarray] = None
    ) -> float:
        """Returns proximity-metric (mean-weighted-distances)"""
        transformed_positions = np.zeros(
            (self.n_samples_, self.n_features_, self.n_clusters)
        )
        bases = np.zeros((self.n_features_, self.n_features_, self.n_clusters))
        distances = np.zeros((self.n_samples_, self.n_clusters))
        self.weights = np.zeros((self.n_samples_, self.n_clusters))

        for kk in range(self.n_clusters):
            bases[:, :, kk] = get_orthogonal_basis(self.cluster_centers_[kk, :])

            transformed_positions[:, :, kk] = map_cartesian_to_infite_stereographic(
                bases[:, :, kk], XX
            )

            distances[:, kk] = np.linalg.norm(
                transformed_positions[:, :, kk]
                - np.tile(self.get_stereographic_center(kk), (self.n_samples_, 1)),
                axis=1,
            )

            if self.distance_type == DistanceWeightType.EXPONENTIAL:
                self.weights[:, kk] = get_exponential_stiffness_weight(
                    self.weights[:, kk], self.stiffness
                )

            elif self.distance_type == DistanceWeightType.QUADRATIC:
                self.weights[:, kk] = get_inverse_square_distance_weight(
                    distances[:, kk]
                )
            else:
                raise ValueError(f"Unkown type {self.distance_type}.")

            if sample_weights is not None:
                self.weights[:, kk] = self.weights[:, kk] * sample_weights
                self.weights[:, kk] = self.weights[:, kk] / np.sum(self.weights[:, kk])
                raise NotImplementedError("Make consistent summing")

        self.labels_ = np.argmin(distances, axis=1)
        mean_squared_distance = 0.0

        # Remove redundant clusters
        it_kk = 0
        it_label = 0
        for _ in range(self.n_clusters):
            ind = self.labels_ == it_kk
            if not np.sum(ind):
                print(f"Removing cluster #{it_label}.")
                ind_large = self.labels_ > it_kk

                self.labels_[ind_large] = self.labels_[ind_large] - 1
                self.cluster_centers_ = np.delete(self.cluster_centers_, it_kk, axis=0)
                self.n_clusters -= 1

            else:
                it_kk += 1

            it_label += 1

        for kk in range(self.n_clusters):
            mapped_center = np.mean(
                transformed_positions[:, :, kk]
                * np.tile(self.weights[:, kk], (self.n_features_, 1)).T,
                axis=0,
            )
            weighted_dist = distances[:, kk] * self.weights[:, kk]
            mean_squared_distance = mean_squared_distance + np.sum(
                weighted_dist * weighted_dist
            )

            self.cluster_centers_[kk, :] = map_infinite_stereographic_to_cartesian(
                bases[:, :, kk], mapped_center.reshape(1, self.n_features_)
            )

            if np.any(np.isnan(self.cluster_centers_)):
                breakpoint()

        return mean_squared_distance / self.n_samples_

    def predict(self, XX: np.ndarray) -> np.ndarray[int]:
        if XX.shape[1] != self.n_features_:
            raise ValueError(f"Wrong data-dimension of {XX.shape}")

        transformed_positions = np.zeros(
            (XX.shape[0], self.n_features_, self.n_clusters)
        )
        distances = np.zeros((XX.shape[0], self.n_clusters))
        bases = np.zeros((self.n_features_, self.n_features_, self.n_clusters))

        for kk in range(self.n_clusters):
            bases[:, :, kk] = get_orthogonal_basis(self.cluster_centers_[kk, :])

            transformed_positions[:, :, kk] = map_cartesian_to_infite_stereographic(
                bases[:, :, kk], XX
            )

            distances[:, kk] = np.linalg.norm(
                transformed_positions[:, :, kk]
                - np.tile(self.get_stereographic_center(kk), (XX.shape[0], 1)),
                axis=1,
            )

        self.labels_ = np.argmin(distances, axis=1)

        return self.labels_


def test_stereographic_mapping():
    n_points = 5
    angle = np.linspace(-np.pi, np.pi, n_points)
    radius = 1.0
    XX = np.vstack((radius * np.cos(angle), radius * np.sin(angle))).T

    base_matrix = get_orthogonal_basis(np.array([1, 0.0]))
    stereographic = map_cartesian_to_infite_stereographic(base_matrix, XX)

    assert np.linalg.norm(stereographic[0, :]) > 1e9
    assert np.linalg.norm(stereographic[-1, :]) > 1e9

    assert np.isclose(stereographic[2, 1], 0), "Only mapped along main-direction."
    assert not np.isclose(stereographic[1, 1], 0)

    assert np.allclose(stereographic[1, 1], -stereographic[-2, 1])
    assert np.allclose(stereographic[:, 0], np.ones(n_points) * radius)

    # Test bijectivity
    XX_restored = map_infinite_stereographic_to_cartesian(base_matrix, stereographic)

    assert np.allclose(XX_restored, XX)


def test_rotated_mapping():
    n_points = 5
    angle = np.linspace(-np.pi, np.pi, n_points)
    radius = 1.0
    XX = np.vstack((radius * np.cos(angle), radius * np.sin(angle))).T

    base_matrix = get_orthogonal_basis(np.array([0, 1.0]))
    stereographic = map_cartesian_to_infite_stereographic(base_matrix, XX)
    assert np.allclose(stereographic[-2, :], [1, 0])
    assert np.linalg.norm(stereographic[1, :]) > 1e6

    # Test bijectivity
    XX_restored = map_infinite_stereographic_to_cartesian(base_matrix, stereographic)
    assert np.allclose(XX_restored, XX)


def test_point_rotated():
    base = np.array([[6.123234e-17, -1.000000e00], [1.000000e00, 6.123234e-17]])
    position = [0.1, 0.9]
    mapped_position = map_cartesian_to_infite_stereographic(base, np.array([position]))

    assert np.allclose(mapped_position, [1.0, 0], atol=0.2)


def test_prediction(visualize=False):
    n_points = 4
    base_radius = 1.0

    angle = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    radii = np.ones(n_points) * base_radius

    XX = np.vstack((radii * np.cos(angle), radii * np.sin(angle))).T

    kmeans = DirectionalKMeans(n_clusters=4)
    kmeans.fit(XX)

    # Set points
    kmeans.cluster_centers_ = copy.deepcopy(XX)

    if visualize:
        x_lim = [-1.5, 1.5]
        y_lim = [-1.5, 1.5]
        nx = ny = 40
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1))).T
        labels = kmeans.predict(positions)

        fig, ax = plt.subplots()
        ax.scatter(XX[:, 0], XX[:, 1], 20.0, marker=".", color="black")
        ax.scatter(0, 0, 50.0, marker="*", color="black")

        cs = ax.contourf(
            x_vals.reshape(nx, ny),
            y_vals.reshape(nx, ny),
            labels.reshape(nx, ny),
            levels=np.arange(kmeans.n_clusters + 1) - 0.5,
            cmap="Accent",
            zorder=-1
            # origin=origin,
        )

        cbar = fig.colorbar(cs)

        ax.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            50,
            marker="x",
            color="black",
        )
        ax.set_aspect("equal", "box")

    position = [0.1, 0.9]
    labels = kmeans.predict(np.array([position]))
    assert labels[0] == 1


def test_circular_fitting(visualize=False):
    n_points = 8
    radius_frequency = 2.0
    base_radius = 1.5

    angle = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    # radii = np.sin(angle * radius_frequency) + base_radius
    radii = np.ones(n_points) * base_radius

    XX = np.vstack((radii * np.cos(angle), radii * np.sin(angle))).T

    kmeans = DirectionalKMeans()
    kmeans.fit(XX)

    if visualize:
        x_lim = [-3, 3]
        y_lim = [-3, 3]
        nx = ny = 40
        x_vals, y_vals = np.meshgrid(
            np.linspace(x_lim[0], x_lim[1], nx), np.linspace(y_lim[0], y_lim[1], ny)
        )
        positions = np.vstack((x_vals.reshape(1, -1), y_vals.reshape(1, -1))).T
        labels = kmeans.predict(positions)

        fig, ax = plt.subplots()
        ax.scatter(XX[:, 0], XX[:, 1], 20.0, marker=".", color="black")
        ax.scatter(0, 0, 50.0, marker="*", color="black")

        cs = ax.contourf(
            x_vals.reshape(nx, ny),
            y_vals.reshape(nx, ny),
            labels.reshape(nx, ny),
            levels=np.arange(kmeans.n_clusters + 1) - 0.5,
            cmap="Accent",
            zorder=-1
            # origin=origin,
        )

        cbar = fig.colorbar(cs)

        ax.scatter(
            kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1],
            50,
            marker="x",
            color="black",
        )

        ax.set_aspect("equal", "box")


if (__name__) == "__main__":
    import matplotlib.pyplot as plt

    # plt.close("all")
    plt.ion()
    # np.set_printoptions(precision=3)

    test_circular_fitting(visualize=True)
    # test_point_rotated()
    # test_rotated_mapping()
    # test_stereographic_mapping()

    # test_prediction(visualize=True)
