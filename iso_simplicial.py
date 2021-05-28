#!/usr/bin/env python3

from generate_quadtree import generate_quadtree
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


bounds = Rect(-6.2, 4.8, -7, 7)
segments = plot_implicit(bounds, fn, gradient)

# format to paste into Desmos to visualize segments
# to paste easily, run the following in dev tools console with the output from this file piped into the clipboard:
# setTimeout(() => {window.navigator.clipboard.readText().then(text => text.split("\n").map((line, i) => Calc.setExpression({id: `clip_${i}`, latex: line, lines: true})))}, 1000)
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

print(
    R"\P_{quad}=\left["
    + ",".join(
        [
            f"({p[0]}, {p[1]})"
            for p in generate_quadtree(bounds, fn, gradient).visualize_borders()
        ]
    )
    + R"\right]"
)
