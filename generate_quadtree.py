"""
5.1. Octree Generation
While our method operates on arbitrary octrees, we provide
a simple method to create an octree that conforms well to the
isosurface using the error metric from Section 3
"""

from quadtree import Quadtree, Rect
from collections import deque
import numpy as np

# arbitrary, can probably be increased
MAX_LEAVES = 10 ** 5
# descend uniformly up to MIN_DEPTH to capture coarse features
MIN_DEPTH = 5
# quantifies maximum acceptable deviation from local linearity
ERROR_THRESHOLD = 10 ** -5
# X_TOLERANCE and Y_TOLERANCE effectively determine the max depth
# TODO: compute X and Y tolerance based on x-width and y-width
#   divided by the screen size (pixels) to exclude details smaller than a pixel
X_TOLERANCE = 0.001
Y_TOLERANCE = 0.001


def should_descend_quadtree(quad: Quadtree, fn, bounds):
    """Assumes quad has dual vertices and values at dual vertices already computed"""
    if quad.depth < MIN_DEPTH:
        # must descend to the minimum depth
        return True
    if quad.width < 10 * X_TOLERANCE or quad.width < 10 * Y_TOLERANCE:
        # descending would create a quad with 5×tolerance dimensions,
        # which would sample points at (average) 2.5×tolerance spacing
        # which captures subpixel details (useless) after apply marching simplices
        return False

    # TODO: maybe use the midpoint + center function values (instead of the dual values)
    #   to test for isoline intersection. This will remove the need to compute duals on
    #   non-leaf quads (saving time) but will reduce the power of the test
    #   (greater chance to not detect an intersection)
    values = np.array(
        [*quad.vertex_values, *quad.edge_dual_values, quad.face_dual_value]
    )
    intersects_isoline = np.any(values > 0) and np.any(values < 0)

    """ We use an error criterion simply
    to avoid refinement in regions of the function that are well
    approximated by a linear function such as flat regions of the
    isosurface."""

    if intersects_isoline and quad.error > ERROR_THRESHOLD:
        return True
    return False


def generate_quadtree(bounds: Rect, fn, gradient):
    """we use a top-down contour-finding approach
    that adds cells to the tree by refining the sampling around
    detected contours"""

    """To begin, we use a uniform refinement
    down to a prescribed level to capture the coarse features of
    the function.

    Beginning with a uniform sampling does guarantee that we
    detect all features greater than the size of a cube in the uni-
    form grid"""

    quadtree = Quadtree(bounds, 0)

    """Then, we analyze F(p) at the dual vertices of
    each cube. We refine each cube that the dual vertices indicate
    the contour intersects until the sum of the errors from Equa-
    tion 1 for each dual vertex in a cube is below a set threshold
    down to a prescribed maximum depth."""

    # we use deque as a FIFO queue
    # (FIFO because we want a breadth of quads instead of a depth of quads in case of exceeding MAX_LEAVES)
    quad_stack = deque([quadtree])

    num_leaves = 1
    while len(quad_stack) > 0 and num_leaves < MAX_LEAVES:
        current_quad = quad_stack.popleft()

        # duals are used:
        #  - in should_descend_quadtree, to test if the isoline crosses through the quad
        #  - in leaf nodes, to compute the segments
        # TODO: precompute the function values at midpoints and center, to save time in child nodes
        current_quad.compute_duals(fn, gradient)
        if should_descend_quadtree(current_quad, fn, bounds):
            current_quad.children = current_quad.generate_children()
            for child in current_quad.children:
                quad_stack.append(child)
            # add 4 for the new quads, subtract 1 for the old quad not being a leaf anymore
            num_leaves += 3
            if num_leaves >= MAX_LEAVES:
                # give up before resolving to the full tolerance
                break
        else:
            # should not descend, so current_quad is a leaf (minimal 2-cell)
            # do nothing for now; just constructing the quadtree
            pass
    # compute duals of leaves in case of breaking early
    for leaf_quad in quad_stack:
        leaf_quad.compute_duals(fn, gradient)

    return quadtree
