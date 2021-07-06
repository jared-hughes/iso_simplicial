import numpy as np
from generate_quadtree import generate_quadtree, Rect
from quadtree import X_TOLERANCE, Y_TOLERANCE


def weighted_intersection(p1, p2):
    """Assuming f(p1)=v1 and f(p2)=v2, perform a linear interpolation
    to find the point pon segment p1--p2 where f(p)=0"""
    v1 = p1[2]
    v2 = p2[2]
    return (v2 * p1 - v1 * p2) / (v2 - v1)


def binary_search_intersection(p1: np.ndarray, p2: np.ndarray, fn):
    if abs(p1[0] - p2[0]) < 2 * X_TOLERANCE and abs(p1[1] - p2[1]) < 2 * Y_TOLERANCE:
        return weighted_intersection(p1, p2)
    else:
        p_mid = (p1 + p2) / 2
        p_mid[2] = fn(p_mid[0], p_mid[1])
        if p_mid[2] == 0:
            return p_mid
        elif np.sign(p_mid[2]) == np.sign(p1[2]):
            return binary_search_intersection(p_mid, p2, fn)
        else:
            return binary_search_intersection(p1, p_mid, fn)


def march_simplex(points: np.ndarray, fn):
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
                binary_search_intersection(
                    points[(i + 1) % 3], points[(i + 2) % 3], fn
                ),
            )
        else:
            i = neg_indices[0][0] if num_neg == 1 else pos_indices[0][0]
            j = (i + 1) % 3
            k = (i + 2) % 3
            yield (
                binary_search_intersection(points[i], points[j], fn),
                binary_search_intersection(points[i], points[k], fn),
            )


def plot_implicit(bounds: Rect, fn):
    """Yields an iterator of line segments (point, point)"""
    quadtree = generate_quadtree(bounds, fn)
    for leaf in quadtree.leaves():
        if not leaf.intersects_isoline:
            continue
        duals = list(leaf.all_duals())
        duals_rotated = [*duals[1:], duals[0]]
        for p2, p3 in zip(duals, duals_rotated):
            yield from march_simplex(np.array([leaf.face_dual, p2, p3]), fn)
