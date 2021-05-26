#!/usr/bin/env python3

from vector import Vector
from quadtree import Rect
from generate_quadtree import generate_quadtree

"""
We make no assumptions about F(p) other than that the
function is piecewise smooth and continuous so that a gra-
dient is well-defined almost everywhere. Places where the
gradient is not defined are sharp features of the function.
"""


def fn(x, y):
    return x ** 2 + 4 * y ** 2 - 2 * y - 2 + x * y


def gradient(x, y):
    return Vector(2 * x + y, 8 * y - 2 + x)


quadtree = generate_quadtree(Rect(-2, 2, -2, 2), fn, gradient, 3)

print(fn(1, 1))
print(gradient(1, 1))
print(quadtree.children[0].boundary)
