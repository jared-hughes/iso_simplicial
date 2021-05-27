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
MAX_QUADS = 10 ** 5
MAX_DEPTH = 10
# quantifies maximum acceptable deviation from local linearity
ERROR_THRESHOLD = 10 ** -5


def generate_quadtree(bounds: Rect, fn, gradient, uniform_depth: int):
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
    initial_leaves = quadtree.subdivide_to_depth(uniform_depth)

    """Then, we analyze F(p) at the dual vertices of
    each cube. We refine each cube that the dual vertices indicate
    the contour intersects until the sum of the errors from Equa-
    tion 1 for each dual vertex in a cube is below a set threshold
    down to a prescribed maximum depth."""

    # we use deque as a FIFO queue
    # (FIFO because we want a breadth of quads instead of a depth of quads in case of exceeding MAX_QUADS)
    quad_stack = deque(initial_leaves)

    num_quads = (len(initial_leaves) * 4 - 1) / 3
    while len(quad_stack) > 0 and num_quads < MAX_QUADS:
        quad = quad_stack.popleft()
        if quad.depth >= MAX_DEPTH:
            # can we `break` instead?
            continue
        quad.compute_duals(fn, gradient)
        values = np.array(
            [*quad.vertex_values, *quad.edge_dual_values, quad.face_dual_value]
        )
        intersects_isoline = np.any(values > 0) and np.any(values < 0)
        """We use an error criterion simply
        to avoid refinement in regions of the function that are well
        approximated by a linear function such as flat regions of the
        isosurface."""
        if intersects_isoline and quad.error > ERROR_THRESHOLD:
            quad.subdivide()
            num_quads += 4
            for child in quad.children:
                quad_stack.append(child)

    return quadtree
