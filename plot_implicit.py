import numpy as np
from generate_quadtree import generate_quadtree
from quadtree import Rect


def weighted_intersection(p1, p2, v1, v2):
    """Assuming f(p1)=v1 and f(p2)=v2, perform a linear interpolation
    to find the point pon segment p1--p2 where f(p)=0"""
    return (v2 * p1 - v1 * p2) / (v2 - v1)


def march_simplex(points, values):
    """Assumes there's no function values 0 (i.e. assumes the isoline does NOT pass through the vertices)
    That would mess with the sign() function, and can cause division by zero if two function values are 0.
    Can be worked around, TODO"""
    num_pos = np.sum(values > 0)
    num_neg = np.sum(values < 0)
    if num_neg > num_pos:
        # negating all values does not affect intersections
        values = -values
    if num_neg != 0 and num_pos != 0:
        # now, two values should be positive and one should be negative
        i = np.where(values < 0)[0][0]
        j = (i + 1) % 3
        k = (i + 2) % 3
        yield (
            weighted_intersection(points[i], points[j], values[i], values[j]),
            weighted_intersection(points[i], points[k], values[i], values[k]),
        )
    else:
        # The function may not pass through this simplex
        # don't yield anything
        pass


def plot_implicit(bounds: Rect, fn):
    """Yields an iterator of line segments (point, point)"""
    quadtree = generate_quadtree(bounds, fn)
    for leaf in quadtree.leaves():
        duals = list(leaf.all_duals())
        duals_rotated = [*duals[1:], duals[0]]
        for (p2, v2), (p3, v3) in zip(duals, duals_rotated):
            yield from march_simplex(
                np.array([leaf.face_dual, p2, p3]),
                np.array([leaf.face_dual_value, v2, v3]),
            )
