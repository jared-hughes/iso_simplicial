import numpy as np
from generate_quadtree import generate_quadtree, Rect


def weighted_intersection(p1, p2):
    """Assuming f(p1)=v1 and f(p2)=v2, perform a linear interpolation
    to find the point pon segment p1--p2 where f(p)=0"""
    v1 = p1[2]
    v2 = p2[2]
    return (v2 * p1 - v1 * p2) / (v2 - v1)


def march_simplex(points: np.ndarray):
    """Assumes there's no function values 0 (i.e. assumes the isoline does NOT pass through the vertices)
    That would mess with the sign() function, and can cause division by zero if two function values are 0.
    Can be worked around, TODO"""
    values = points[:, 2]
    zero_indices = np.where(values == 0)
    num_zeros = np.size(zero_indices)
    pos_indices = np.where(values > 0)
    num_pos = np.size(pos_indices)
    neg_indices = np.where(values < 0)
    num_neg = np.size(neg_indices)
    if num_zeros >= 2:
        # num_zeros == 2 handles the case where the isoline passes through two vertices
        # no unique answer for num_zeros == 3, but we handle it here to avoid division by 0
        yield (points[zero_indices[0][0]], points[zero_indices[0][1]])
    elif num_pos > 0 and num_neg > 0:
        if num_zeros == 1:
            i = zero_indices[0][0]
            yield (
                points[i],
                weighted_intersection(points[(i + 1) % 3], points[(i + 2) % 3]),
            )
        else:
            i = neg_indices[0][0] if num_neg == 1 else pos_indices[0][0]
            j = (i + 1) % 3
            k = (i + 2) % 3
            yield (
                weighted_intersection(points[i], points[j]),
                weighted_intersection(points[i], points[k]),
            )


def plot_implicit(bounds: Rect, fn):
    """Yields an iterator of line segments (point, point)"""
    quadtree = generate_quadtree(bounds, fn)
    for leaf in quadtree.leaves():
        duals = list(leaf.all_duals())
        duals_rotated = [*duals[1:], duals[0]]
        for p2, p3 in zip(duals, duals_rotated):
            yield from march_simplex(
                np.array([leaf.face_dual, p2, p3]),
            )
