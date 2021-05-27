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


class Quadtree:
    def __init__(self, left, right, bottom, top, depth: int):
        """+x to the right, +y to the top, so right>left and top>bottom"""
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.depth = depth
        self.children: List[Quadtree] = []

    def subdivide(self):
        """Uniform subdivision"""
        mx = (self.left + self.right) / 2
        my = (self.top + self.bottom) / 2
        next_depth = self.depth + 1
        self.children = [
            Quadtree(self.left, mx, my, self.top, next_depth),
            Quadtree(mx, self.right, my, self.top, next_depth),
            Quadtree(mx, self.right, self.bottom, my, next_depth),
            Quadtree(self.left, mx, self.bottom, my, next_depth),
        ]

    def subdivide_to_depth(self, depth: int, fn, gradient):
        """Perform uniform subdivision until the quadtree reaches a depth of `depth`"""
        if depth > self.depth:
            self.subdivide()
            for quad in self.children:
                quad.subdivide_to_depth(depth, fn, gradient)
        # in the future, only compute duals for minimal cells, i.e
        # don't compute a dual on the right side of a quad if the quad to its right
        # is subdivided
        self.compute_duals(fn, gradient)

    def shrunk_bounds(self):
        # any number > 0 should work to avoid degenerate triangles
        BOUND_EPSILON = 0.01
        return BOUND_EPSILON * np.array(
            [self.right, self.left, self.top, self.bottom]
        ) + (1 - BOUND_EPSILON) * np.array(
            [self.left, self.right, self.bottom, self.top]
        )

    def compute_edge_duals(self, vertices_3d, normals):
        # Could probably use scipy.optimize here like in compute_face_dual
        # Though it might be slower than just solving for the intersection
        # Pro of scipy.optimize: it would find the minimum inside the bounds
        # instead of finding the minimum, then clipping to the bounds
        x_min, x_max, y_min, y_max = self.shrunk_bounds()
        edge_duals_horiz = [
            [
                np.clip(
                    intersect_normals(
                        extract_xz(vertices_3d[i]),
                        extract_xz(vertices_3d[j]),
                        extract_xz(normals[i]),
                        extract_xz(normals[j]),
                    ),
                    x_min,
                    x_max,
                ),
                vertices_3d[i][1],
            ]
            for i, j in [(0, 1), (2, 3)]
        ]
        edge_duals_vert = [
            [
                vertices_3d[i][0],
                np.clip(
                    intersect_normals(
                        extract_yz(vertices_3d[i]),
                        extract_yz(vertices_3d[j]),
                        extract_yz(normals[i]),
                        extract_yz(normals[j]),
                    ),
                    y_min,
                    y_max,
                ),
            ]
            for i, j in [(1, 2), (3, 0)]
        ]
        # clockwise from the top edge
        return [
            edge_duals_horiz[0],
            edge_duals_vert[0],
            edge_duals_horiz[1],
            edge_duals_vert[1],
        ]

    def compute_face_dual(self, vertices_3d, normals):
        """The normals in XYZ, coupled with the vertices_3d, define four hyperplanes. The goal is to find
        the point that minimizes the sum of squared distance from the four hyperplanes"""
        B = [np.dot(n, p) for n, p in zip(normals, vertices_3d)]

        x_min, x_max, y_min, y_max = self.shrunk_bounds()
        result = scipy.optimize.lsq_linear(
            normals, B, bounds=([x_min, y_min, -np.inf], [x_max, y_max, np.inf])
        )
        # (note for later: result.cost and result.optimality)
        # might want to consider result.success
        return result.x

    def compute_duals(self, fn, gradient):
        """Insert dual vertices. In the future, this should only by done for minimal cells
        and should avoid duplicating calculation of `gradient` over the same points for vertices
        shared between two edges"""
        # precalculations
        vertices = [
            [self.left, self.top],
            [self.right, self.top],
            [self.right, self.bottom],
            [self.left, self.bottom],
        ]
        values = [fn(v[0], v[1]) for v in vertices]
        gradients = [gradient(v[0], v[1]) for v in vertices]
        normals = [np.array([g[0], g[1], -1]) for g in gradients]
        normals = np.array([n / np.linalg.norm(n) for n in normals])
        vertices3d = np.array(
            [[v[0], v[1], value] for v, value in zip(vertices, values)]
        )
        self.edge_duals = self.compute_edge_duals(vertices3d, normals)
        self.face_dual = self.compute_face_dual(vertices3d, normals)

    def __str__(self):
        return f"Quadtree[depth={self.depth}, {self.left}≤x≤{self.right}, {self.bottom}≤y≤{self.top}, children:{len(self.children)}]"
