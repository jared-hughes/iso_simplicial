#!/usr/bin/env python3

import numpy as np
from generate_quadtree import generate_quadtree

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


quadtree = generate_quadtree(-2, 2, -2, 2, fn, gradient, 3)

print(fn(1, 1))
print(gradient(1, 1))
quad = quadtree.children[2].children[1]
print(quad)
print(np.round(quad.edge_duals, 4))
print(np.round(quad.face_dual, 4))
