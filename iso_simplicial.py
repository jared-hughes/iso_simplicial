#!/usr/bin/env python3

import numpy as np
from plot_implicit import plot_implicit
from quadtree import Rect

"""
We make no assumptions about F(p) other than that the
function is piecewise smooth and continuous so that a gra-
dient is well-defined almost everywhere. Places where the
gradient is not defined are sharp features of the function.
"""


def fn(x, y):
    return np.sin(x) + np.sin(y) + 0.1 * x - 0.5


def gradient(x, y):
    # Could always be lazy and just do finite differences.
    # I'm sure numpy has something built-in that uses some limiting process.
    # Analytic is faster though
    return np.array([np.cos(x) + 0.1, np.cos(y)])


segments = plot_implicit(Rect(-6, 5, -7, 7), fn, gradient)

# format to paste into Desmos to visualize segments
print(
    R"\P_{iso}=\left["
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
