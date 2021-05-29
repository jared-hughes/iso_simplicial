from typing import List, Callable
import numpy as np

CYCLIC_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0)]


def nonlinearity_along_edge(p1, p2, fn):
    midpoint = (p1 + p2) / 2
    return np.abs(p1[2] - 2 * fn(midpoint[0], midpoint[1]) + p2[2]) / np.max(
        np.abs([p1[2], p2[2]])
    )


def get_dual_point_between(p1, p2, fn):
    """Assumes the value of the points are not NaN/Infinity and
    they are stored as [x, y, f(x,y)]"""
    if np.sign(p1[2]) != np.sign(p2[2]):
        # the isoline passes somewhere between p1 and p2, so place a dual point mid-way
        # Maybe doing a lerp of projected gradients would be better
        return 0.5 * (p1 + p2)
    else:
        eps = 0.001
        nearby_1 = (1 - eps) * p1 + eps * p2
        nearby_2 = (1 - eps) * p2 + eps * p1
        # (finite difference) directional derivative in the drection p1→p2, without the 1/eps factor
        ddt_1 = fn(nearby_1[0], nearby_1[1]) - p1[2]
        # (finite difference) directional derivative in the drection p2→p1, without the 1/eps factor
        ddt_2 = fn(nearby_2[0], nearby_2[1]) - p2[2]
        if np.sign(ddt_1) == np.sign(ddt_2):
            # signs are the same, so the corresponding lines would not intersect between the two points
            # just take midpoint
            return 0.5 * (p1 + p2)
        else:
            return lerpByZ(
                np.array([p1[0], p1[1], ddt_1]), np.array([p2[0], p2[1], ddt_2])
            )


def lerpByZ(p1, p2):
    """Find the point along the segment p1--p2 where z=0"""
    if np.sign(p1[2]) == np.sign(p2[2]):
        # give up, but don't throw an error
        return (p1 + p2) / 2
    return (p2[2] * p1 - p1[2] * p2) / (p2[2] - p1[2])


class Quadtree:
    def __init__(
        self,
        vertices: List[np.ndarray],
        depth: int,
        fn: Callable[[float, float], float],
    ):
        """Vertices are a list of (x, y, f(x,y)) values"""
        self.vertices = vertices
        self.depth = depth
        self.fn = fn
        self.children: List[Quadtree] = []

    def _apply_func_to(self, p: np.ndarray):
        """Mutates p"""
        p[2] = self.fn(p[0], p[1])
        return p

    def apply_func_to_vertices(self):
        for v in self.vertices:
            self._apply_func_to(v)

    def _construct_point(self, x: float, y: float):
        return np.array([x, y, self.fn(x, y)])

    def compute_midpoints(self):
        center = 0.5 * (self.vertices[0] + self.vertices[2])
        self.edge_midpoints = (
            self._construct_point(center[0], self.vertices[0][1]),
            self._construct_point(self.vertices[1][0], center[1]),
            self._construct_point(center[0], self.vertices[2][1]),
            self._construct_point(self.vertices[0][0], center[1]),
        )
        self.center_midpoint = self._construct_point(center[0], center[1])

    def compute_children(self):
        """Uniform subdivision"""
        self.compute_midpoints()
        next_depth = self.depth + 1
        self.children = [
            Quadtree(
                (
                    self.vertices[0],
                    self.edge_midpoints[0],
                    self.center_midpoint,
                    self.edge_midpoints[3],
                ),
                next_depth,
                self.fn,
            ),
            Quadtree(
                (
                    self.edge_midpoints[0],
                    self.vertices[1],
                    self.edge_midpoints[1],
                    self.center_midpoint,
                ),
                next_depth,
                self.fn,
            ),
            Quadtree(
                (
                    self.center_midpoint,
                    self.edge_midpoints[1],
                    self.vertices[2],
                    self.edge_midpoints[2],
                ),
                next_depth,
                self.fn,
            ),
            Quadtree(
                (
                    self.edge_midpoints[3],
                    self.center_midpoint,
                    self.edge_midpoints[2],
                    self.vertices[3],
                ),
                next_depth,
                self.fn,
            ),
        ]

    def _get_edge_duals(self):
        return [
            self._apply_func_to(
                get_dual_point_between(self.vertices[i], self.vertices[j], self.fn)
            )
            for i, j in CYCLIC_PAIRS
        ]

    def _get_face_dual(self):
        # return the center between two diagonal vertices as long as the function value
        # at the center has a different function sign than the two diagonals
        a, b, c, d = self.vertices
        if np.sign(a[2]) == np.sign(c[2]):
            center = get_dual_point_between(a, c, self.fn)
            if np.sign(self.fn(center[0], center[1])) != np.sign(a[2]):
                return center
        if np.sign(b[2]) == np.sign(d[2]):
            center = get_dual_point_between(b, d, self.fn)
            if np.sign(self.fn(center[0], center[1])) != np.sign(b[2]):
                return center
        return self._apply_func_to((a + c) / 2)

    def compute_duals(self):
        """Insert dual vertices. In the future, this should only be done for minimal cells"""
        self.edge_duals = self._get_edge_duals()
        self.face_dual = self._get_face_dual()
        self.nonlinearity = np.max(
            [
                nonlinearity_along_edge(self.vertices[i], self.vertices[j], self.fn)
                for i, j in CYCLIC_PAIRS + [(0, 2), (1, 3)]
            ]
        )

    """ Following are debug functions """

    def _directional_duals(self, direction: int):
        """All edge duals, including children, in clockwise order as an iterator
        of tuples (xy position, value)
        Direction: 0=top, 1=right, 2=bottom, 3=left"""
        if len(self.children) != 0:
            yield from self.children[direction]._directional_duals(direction)
        yield self.edge_duals[direction]
        if len(self.children) != 0:
            yield from self.children[(direction + 1) % 4]._directional_duals(direction)

    def all_duals(self):
        """All edge duals and vertices, including children, in clockwise order"""
        for i in range(4):
            yield self.vertices[i]
            yield from self._directional_duals(i)

    def leaves(self):
        if len(self.children) == 0:
            yield self
        for child in self.children:
            yield from child.leaves()

    def __str__(self):
        return f"Quadtree[depth={self.depth}, {self.left}≤x≤{self.right}, {self.bottom}≤y≤{self.top}, children:{len(self.children)}]"

    def visualize_borders(self):
        """Returns a list of points [x: str | float, y: str | float] which draw the quadtree when connected in Desmos"""
        if len(self.children) > 0:
            yield self.edge_midpoints[0]
            yield self.edge_midpoints[2]
            yield ["0/0", "0"]
            yield self.edge_midpoints[3]
            yield self.edge_midpoints[1]
            yield ["0/0", "0"]
        for child in self.children:
            yield from child.visualize_borders()

    def leaf_all_duals(self):
        if len(self.children) == 0:
            yield from self.all_duals()
        for child in self.children:
            yield from child.leaf_all_duals()

    def leaf_face_duals(self):
        if len(self.children) == 0:
            yield self.face_dual
        for child in self.children:
            yield from child.leaf_face_duals()

    @property
    def width(self):
        return self.vertices[1][0] - self.vertices[0][0]

    @property
    def height(self):
        return self.vertices[1][1] - self.vertices[2][1]
