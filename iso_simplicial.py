#!/usr/bin/env python3

from generate_quadtree import generate_quadtree
import numpy as np
from plot_implicit import plot_implicit
from quadtree import Rect
import cProfile, pstats

"""
We make no assumptions about F(p) other than that the
function is piecewise smooth and continuous so that a gra-
dient is well-defined almost everywhere. Places where the
gradient is not defined are sharp features of the function.
"""

examples = [
    [lambda x, y: y - np.sin(5 * x), Rect(-2, 2, -1.5, 1.5)],
    [lambda x, y: x * x + y * y - 4, Rect(-3, 3, -3, 3)],
    [lambda x, y: y * (x - y) ** 2 - 4 * x - 8, Rect(-6, 8, -6, 6)],
    [lambda x, y: y ** 2 - x ** 3 + x, Rect(-2, 2, -2, 2)],
    [
        lambda x, y: x ** 2
        + 4 * y ** 2
        - 4
        + x * y
        + 5 * np.sin(5 * x)
        + 5 * np.sin(5 * y),
        Rect(-4, 4, -2, 2),
    ],
    [lambda x, y: np.tan(x * y), Rect(-5, 5, -5, 5)],
]

fn, bounds = examples[0]

profiler = cProfile.Profile()
profiler.enable()
segments = list(plot_implicit(bounds, fn))
profiler.disable()
with open("profile.log", "w") as stream:
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumtime")
    stats.print_stats()

# format to paste into Desmos to visualize segments
# to paste easily, run the following in dev tools console with the output from this file piped into the clipboard:
# setTimeout(() => {window.navigator.clipboard.readText().then(text => text.split("\n").map((line, i) => Calc.setExpression({id: `clip_${i}`, latex: line, lines: i < 2})))}, 1000)
# Use https://gist.github.com/jared-hughes/1bab5d94e2ad0ab326180a21e3f955c0 to bypass Desmos's 10000 point limit
def pointLatex(p):
    return f"({p[0]},{p[1]})"


def listLatex(L):
    return R"\left[" + ",".join(L) + R"\right]"


def pointListLatex(L):
    return listLatex(map(pointLatex, L))


print(
    R"\P_{iso}="
    + listLatex(
        f"{pointLatex(segment[0])},{pointLatex(segment[1])},(0/0,0/0)"
        for segment in segments
    )
)

quadtree = generate_quadtree(bounds, fn)

print(R"\P_{quad}=" + pointListLatex(quadtree.visualize_borders()))
print(R"\P_{faceDuals}=" + pointListLatex(quadtree.leaf_face_duals()))
print(R"\P_{edgeDuals}=" + pointListLatex(quadtree.leaf_all_duals()))
