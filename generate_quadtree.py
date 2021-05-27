"""
5.1. Octree Generation
While our method operates on arbitrary octrees, we provide
a simple method to create an octree that conforms well to the
isosurface using the error metric from Section 3
"""

from quadtree import Quadtree


def generate_quadtree(left, right, bottom, top, fn, gradient, uniform_depth: int = 3):
    """we use a top-down contour-finding approach
    that adds cells to the tree by refining the sampling around
    detected contours"""

    """To begin, we use a uniform refinement
    down to a prescribed level to capture the coarse features of
    the function.

    Beginning with a uniform sampling does guarantee that we
    detect all features greater than the size of a cube in the uni-
    form grid"""

    quadtree = Quadtree(left, right, bottom, top, 0)
    quadtree.subdivide_to_depth(uniform_depth, fn, gradient)

    return quadtree

    """ TODO """

    """Then, we analyze F(p) at the dual vertices of
    each cube. We refine each cube that the dual vertices indicate
    the contour intersects until the sum of the errors from Equa-
    tion 1 for each dual vertex in a cube is below a set threshold
    down to a prescribed maximum depth."""

    """We use an error criterion simply
    to avoid refinement in regions of the function that are well
    approximated by a linear function such as flat regions of the
    isosurface."""
