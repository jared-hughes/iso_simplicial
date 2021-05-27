#!/usr/bin/env python3

import numpy as np
from plot_implicit import plot_implicit

"""
We make no assumptions about F(p) other than that the
function is piecewise smooth and continuous so that a gra-
dient is well-defined almost everywhere. Places where the
gradient is not defined are sharp features of the function.
"""


def fn(x, y):
    # Throwing in the sines to be able to test edge duals that are not just centered
    return x ** 2 + 4 * y ** 2 - 4 + x * y + 2 * np.sin(x) + 0.5 * np.sin(5 * y)


def gradient(x, y):
    # Could always be lazy and just do finite differences.
    # I'm sure numpy has something built-in that uses some limiting process.
    # Analytic is faster though
    return np.array([2 * x + y + 2 * np.cos(x), 8 * y + x + 2.5 * np.cos(5 * y)])


segments = plot_implicit(-3, 2, -2, 2, fn, gradient)

# format to paste into Desmos to visualize segments
print(
    R"P_{iso}=\left["
    + ",".join(
        [
            f"({segment[0][0]},{segment[0][1]}),"
            + f"({segment[1][0]},{segment[1][1]}),"
            + "(0/0,0/0)"
            for segment in segments
        ]
    )
    + R"\right]"
)
