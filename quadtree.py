from typing import List
from dataclasses import dataclass
import numpy as np
import scipy.optimize


@dataclass
class Rect:
    left: float
    right: float
    bottom: float
    top: float

    @property
    def width(self):
        return self.right - self.left

    @property
    def height(self):
        return self.top - self.bottom

    def shrunk_by(self, e):
        """e = proportion shrink. Any number > 0 should work to avoid degenerate triangles"""
        return Rect(
            e * self.right + (1 - e) * self.left,
            e * self.left + (1 - e) * self.right,
            e * self.top + (1 - e) * self.bottom,
            e * self.bottom + (1 - e) * self.top,
        )


def extract_xy(p):
    return np.array([p[0], p[1]])


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
    """Find the point along the line p1←→p2 where z=0"""
    return extract_xy((p2[2] * p1 - p1[2] * p2) / (p2[2] - p1[2]))


class Quadtree(Rect):
    def __init__(self, bounds: Rect, depth: int):
        """+x to the right, +y to the top, so right>left and top>bottom"""
        super().__init__(bounds.left, bounds.right, bounds.bottom, bounds.top)
        self.depth = depth
        self.children: List[Quadtree] = []

    def generate_children(self):
        """Uniform subdivision"""
        mx = (self.left + self.right) / 2
        my = (self.top + self.bottom) / 2
        next_depth = self.depth + 1
        return [
            Quadtree(Rect(self.left, mx, my, self.top), next_depth),
            Quadtree(Rect(mx, self.right, my, self.top), next_depth),
            Quadtree(Rect(mx, self.right, self.bottom, my), next_depth),
            Quadtree(Rect(self.left, mx, self.bottom, my), next_depth),
        ]

    def compute_edge_duals(self, vertices_3d, fn, shrunk_region):
        return [
            extract_xy(get_dual_point_between(vertices_3d[i], vertices_3d[j], fn))
            for i, j in [(0, 1), (1, 2), (2, 3), (3, 0)]
        ]

    def compute_face_dual(self, vertices_3d, fn, shrunk_region):
        # return the center between two diagonal vertices as long as the function value
        # at the center has a different function sign than the two diagonals
        a, b, c, d = vertices_3d
        if np.sign(a[2]) == np.sign(c[2]):
            center = get_dual_point_between(a, c, fn)
            if np.sign(fn(center[0], center[1])) != np.sign(a[2]):
                return center
        if np.sign(b[2]) == np.sign(d[2]):
            center = get_dual_point_between(b, d, fn)
            if np.sign(fn(center[0], center[1])) != np.sign(b[2]):
                return center
        return (a + c) / 2

    def compute_duals(self, fn):
        """Insert dual vertices. In the future, this should only by done for minimal cells
        and should avoid duplicating calculation of `gradient` over the same points for vertices
        shared between two edges"""
        # precalculations
        self.vertices = [
            [self.left, self.top],
            [self.right, self.top],
            [self.right, self.bottom],
            [self.left, self.bottom],
        ]
        values = [fn(v[0], v[1]) for v in self.vertices]
        self.vertex_values = values
        vertices3d = np.array(
            [[v[0], v[1], value] for v, value in zip(self.vertices, values)]
        )
        shrunk_region = self.shrunk_by(0.01)
        self.edge_duals = self.compute_edge_duals(vertices3d, fn, shrunk_region)
        self.edge_dual_values = [fn(v[0], v[1]) for v in self.edge_duals]
        fd = self.compute_face_dual(vertices3d, fn, shrunk_region)
        self.face_dual = fd[0:2]
        self.face_dual_value = fn(self.face_dual[0], self.face_dual[1])

    def directional_duals(self, direction: int):
        """All edge duals, including children, in clockwise order as an iterator
        of tuples (xy position, value)
        Direction: 0=top, 1=right, 2=bottom, 3=left"""
        if len(self.children) != 0:
            yield from self.children[direction].directional_duals(direction)
        yield (self.edge_duals[direction], self.edge_dual_values[direction])
        if len(self.children) != 0:
            yield from self.children[(direction + 1) % 4].directional_duals(direction)

    def all_duals(self):
        """All duals, including children, in clockwise order as an iterator of
        tuples (xy position, value)"""
        for i in range(4):
            yield (self.vertices[i], self.vertex_values[i])
            yield from self.directional_duals(i)

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
            mx = (self.left + self.right) / 2
            my = (self.top + self.bottom) / 2
            yield [mx, self.top]
            yield [mx, self.bottom]
            yield ["0/0", "0"]
            yield [self.left, my]
            yield [self.right, my]
            yield ["0/0", "0"]
        for child in self.children:
            yield from child.visualize_borders()
