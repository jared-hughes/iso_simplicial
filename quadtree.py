from typing import List
from dataclasses import dataclass
import numpy as np
import scipy.optimize


def intersect_normals(p, q, n, m):
    """Given vertices p,q and normals n,m, return the intersection coordinate between the lines passing
    through the vertices and perpendicular to the normals in XZ or YZ space. This coordinate would
    need to be combined with the other coordinate back in XY space."""
    return q[0] - m[1] * np.dot(n, q - p) / np.linalg.det([n, m])


def extract_xz(p):
    return np.array([p[0], p[2]])


def extract_yz(p):
    return np.array([p[1], p[2]])


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

    def compute_edge_duals(self, vertices_3d, normals, shrunk_region):
        # Could probably use scipy.optimize here like in compute_face_dual
        # Though it might be slower than just solving for the intersection
        # Pro of scipy.optimize: it would find the minimum inside the bounds
        # instead of finding the minimum, then clipping to the bounds
        edge_duals_horiz = [
            np.array(
                [
                    np.clip(
                        intersect_normals(
                            extract_xz(vertices_3d[i]),
                            extract_xz(vertices_3d[j]),
                            extract_xz(normals[i]),
                            extract_xz(normals[j]),
                        ),
                        shrunk_region.left,
                        shrunk_region.right,
                    ),
                    vertices_3d[i][1],
                ]
            )
            for i, j in [(0, 1), (2, 3)]
        ]
        edge_duals_vert = [
            np.array(
                [
                    vertices_3d[i][0],
                    np.clip(
                        intersect_normals(
                            extract_yz(vertices_3d[i]),
                            extract_yz(vertices_3d[j]),
                            extract_yz(normals[i]),
                            extract_yz(normals[j]),
                        ),
                        shrunk_region.bottom,
                        shrunk_region.top,
                    ),
                ]
            )
            for i, j in [(1, 2), (3, 0)]
        ]
        # clockwise from the top edge
        return [
            edge_duals_horiz[0],
            edge_duals_vert[0],
            edge_duals_horiz[1],
            edge_duals_vert[1],
        ]

    def compute_face_dual(self, vertices_3d, normals, shrunk_region):
        """The normals in XYZ, coupled with the vertices_3d, define four hyperplanes. The goal is to find
        the point that minimizes the sum of squared distance from the four hyperplanes"""
        B = [np.dot(n, p) for n, p in zip(normals, vertices_3d)]

        result = scipy.optimize.lsq_linear(
            normals,
            B,
            bounds=(
                [shrunk_region.left, shrunk_region.bottom, -np.inf],
                [shrunk_region.right, shrunk_region.top, np.inf],
            ),
        )
        return result.x

    def compute_duals(self, fn, gradient):
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
        gradients = [gradient(v[0], v[1]) for v in self.vertices]
        normals = [np.array([g[0], g[1], -1]) for g in gradients]
        normals = np.array([n / np.linalg.norm(n) for n in normals])
        vertices3d = np.array(
            [[v[0], v[1], value] for v, value in zip(self.vertices, values)]
        )
        shrunk_region = self.shrunk_by(0.01)
        self.edge_duals = self.compute_edge_duals(vertices3d, normals, shrunk_region)
        self.edge_dual_values = [fn(v[0], v[1]) for v in self.edge_duals]
        fd = self.compute_face_dual(vertices3d, normals, shrunk_region)
        self.face_dual = fd[0:2]
        self.face_dual_value = fn(self.face_dual[0], self.face_dual[1])
        self.error = np.abs(self.face_dual_value - fd[2])

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
