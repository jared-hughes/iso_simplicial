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
    return x ** 2 - y ** 2 - np.log(np.abs(x))


bounds = Rect(-6.2, 4.8, -7, 7)
segments = plot_implicit(bounds, fn)

# format to paste into Desmos to visualize segments
# to paste easily, run the following in dev tools console with the output from this file piped into the clipboard:
# setTimeout(() => {window.navigator.clipboard.readText().then(text => text.split("\n").map((line, i) => Calc.setExpression({id: `clip_${i}`, latex: line, lines: true})))}, 1000)
# Use https://gist.github.com/jared-hughes/1bab5d94e2ad0ab326180a21e3f955c0 to bypass Desmos's 10000 point limit
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
        [f"({p[0]}, {p[1]})" for p in generate_quadtree(bounds, fn).visualize_borders()]
    )
    + R"\right]"
)
