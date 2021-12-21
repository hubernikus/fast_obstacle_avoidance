"""
The :mod:`collision_avoider` module implements various types of obstacles.
"""

from ._base import SingleModulationAvoider
from .obstacle_avoider import FastObstacleAvoider
from .lidar_avoider import FastLidarAvoider

__all__ = [
    "SingleModulationAvoider",
    "FastObstacleAvoider",
    "FastLidarAvoider"
    ]
