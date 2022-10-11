"""
Comparsion simple / setup.
"""

import math
import numpy as np

from fast_obstacle_avoidance.comparison.vfh_avoider_m import controllerVFH


def test_simple_setup():
    angles = np.linspace(-math.pi / 2, math.pi / 2, 10)
    ranges = 2 - np.cos(angles)
    ranges = 0.5 * ranges

    input_direction = 0.1

    vfh = controllerVFH(NumAngularSectors=20, HistogramThresholds=(1, 2))

    # vfh.RobotRadius = vfh_options.RobotRadius;

    output_direction = vfh(ranges, angles, input_direction)

    print(output_direction)

    # Compare to real MATLAB output
    assert np.isclose(2.1279, output_direction)


if (__name__) == "__main__":
    test_simple_setup()
