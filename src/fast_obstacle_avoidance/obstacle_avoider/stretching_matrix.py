"""
Simplified obstacle avoidance for mobile for mobile robots based on
DS & reference_direction

The algorithm is developed for shared control,
> human will know the best path, just be fast and don't get stuck
(make unexpected mistakes)
"""
# Use python 3.10 [annotations / typematching]
from __future__ import annotations  # Not needed from python 3.10 onwards

import warnings
from abc import ABC, abstractmethod

import numpy as np
from numpy import linalg as LA


class StretchingMatrixFunctor(ABC):
    def __init__(self, free_tail_flow=True, power_weights=0.2):
        self.free_tail_flow = free_tail_flow
        self.power_weights = power_weights

    def get_weight_vel(self, reference_dir, velocity):
        """Returns velocity weight weight(<reference_dir, velocity>) [-1, 1] -> [0, 1]"""
        if not LA.norm(reference_dir) or not LA.norm(velocity):
            return 0

        return (
            np.maximum(
                0.0,
                (
                    np.dot(reference_dir, velocity)
                    / (LA.norm(reference_dir) * LA.norm(velocity))
                ),
            )
            ** self.power_weights
        )

    def get_weight_importance(self, importance_variable):
        """Returns importance weight [0, infty] -> [1, 0]"""
        return np.minimum(1.0, 1.0 / importance_variable)

    def get_free_tail_flow_lambdas(
        self, lambda_tang, lambda_ref, w_importance, w_velocity
    ):
        lambda_tang_free = (
            w_importance * w_velocity + (1 - w_importance * w_velocity) * lambda_tang
        )

        if w_velocity:
            lambda_ref_free = (
                w_importance * lambda_tang_free + (1 - w_importance) * lambda_ref
            )
        else:
            lambda_ref_free = lambda_tang_free

        return lambda_ref_free, lambda_ref_free

    def get(
        self,
        importance_variable: float,
        reference_direction: np.ndarray,
        normal_direction: np.ndarray = None,
        initial_velocity: np.ndarray = None,
    ) -> np.ndarray:

        if normal_direction is None:
            normal_direction = reference_direction

        weight_vel = self.get_weight_vel(reference_direction, initial_velocity)
        weight_importance = self.get_weight_importance(importance_variable)

        lambda_ref, lambda_tang = self.get_lambda_weights(
            importance_variable, weight_vel
        )

        weight_normvel = np.maximum(
            0, np.dot(normal_direction, initial_velocity)
        ) / LA.norm(initial_velocity)

        if self.free_tail_flow and weight_normvel:
            # breakpoint()
            lambda_ref, lambda_tang = self.get_free_tail_flow_lambdas(
                lambda_tang=lambda_tang,
                lambda_ref=lambda_ref,
                w_importance=weight_importance,
                w_velocity=weight_normvel,
            )

        stretching_vector = np.hstack(
            (lambda_ref, lambda_tang * np.ones(reference_direction.shape[0] - 1))
        )

        return np.diag(stretching_vector)

    @abstractmethod
    def get_lambda_weights(self, weight, weight_vel):
        pass


class StretchingMatrixBasic(StretchingMatrixFunctor):
    def __init__(self, free_tail_flow: bool = True, pow_fact: float = 2):
        self.free_tail_flow = free_tail_flow

        # Define lambda-variable paramters
        self.pow_fact = pow_fact

    def get_lambda_weights(self, weight, weight_vel):
        lambda_ref = (1 - weight) ** self.pow_fact * np.sign(weight_vel)
        lambda_tang = np.min((1 + weight) ** self.pow_fact, 2)

        return lambda_ref, lambda_tang


class StretchingMatrixExponential(StretchingMatrixFunctor):
    def __init__(
        self,
        power_ref: float = 1,
        const_ref: float = 0.1,
        power_tang: float = 1,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Define lambda-variable paramteres
        self.power_ref = power_ref
        self.const_ref = const_ref

        self.power_tang = power_tang

    def get_lambda_weights(self, weight, weight_vel):
        lambda_tang = np.exp((-1) * self.const_ref * weight ** self.power_ref)
        lambda_ref = np.exp((-1) * np.log(2) * (weight ** self.power_tang - 1)) - 1
        if weight_vel < 0 and weight > 1:
            lambda_ref = (-1) * lambda_ref

        return lambda_ref, lambda_tang


class StretchingMatrixTrigonometric(StretchingMatrixFunctor):
    # def __init__(self,*args, **kwargs):
    # super().__init__(*args, **kwargs)

    def get_lambda_weights(self, weight, weight_vel):
        if weight < 1:
            lambda_tang = 1 + np.sin(np.pi /2 *weight)
        else:
            lambda_tang = 2 * np.sin(np.pi / (2*weight))

        if weight < 2:
            lambda_ref = np.cos(np.pi/2*weight)
        else:
            lambda_ref = -1

            
        # if weight < 2:
            # lambda_tang = 1 + np.sin(np.pi / 2 * weight)
            # lambda_ref = np.cos(np.pi / 2 * weight)

        # else:
            # lambda_tang = 1
            # lambda_ref = -1

        if weight_vel < 0 and weight > 1:
            lambda_ref = (-1) * lambda_ref

        return lambda_ref, lambda_tang
