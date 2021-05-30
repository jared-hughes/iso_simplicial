from typing import List, Callable
import numpy as np
import scipy.optimize

CYCLIC_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 0)]
SHRINK_FACTOR = 0.05
# X_TOLERANCE and Y_TOLERANCE effectively determine the max depth
# TODO: compute X and Y tolerance based on x-width and y-width
#   divided by the screen size (pixels) to exclude details smaller than a pixel
X_TOLERANCE = 0.005
Y_TOLERANCE = 0.005
GRADIENT_EPS_X = X_TOLERANCE / 10
GRADIENT_EPS_Y = Y_TOLERANCE / 10


def nonlinearity_along_edge(p1, p2, fn):
    midpoint = (p1 + p2) / 2
    return np.abs(p1[2] - 2 * fn(midpoint[0], midpoint[1]) + p2[2]) / np.max(
        np.abs([p1[2], p2[2]])
    )


def lerpByZ(p1, p2):
    """Find the point along the segment p1--p2 where z=0"""
    if np.sign(p1[2]) == np.sign(p2[2]):
        # give up, but don't throw an error
        return (p1 + p2) / 2
    return (p2[2] * p1 - p1[2] * p2) / (p2[2] - p1[2])


def extract_coord(p, axis):
    return np.array([p[axis], p[2]])


def intersect_normals(p, q, n, m):
    """Given 2D vertices p,q and normals n,m, return the intersection coordinate between the lines passing
    through the vertices and perpendicular to the normals in XZ or YZ space. This coordinate would
    need to be combined with the other coordinate back in XY space."""
    return q[0] - m[1] * np.dot(n, q - p) / np.linalg.det([n, m])


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

    def _get_edge_dual_between(self, i: int, j: int):
        axis = i % 2
        p1 = self.vertices[i]
        p2 = self.vertices[j]
        # if np.sign(p1[2]) != np.sign(p2[2]):
        #     # the isoline passes somewhere between p1 and p2, so place a dual point mid-way
        #     # Maybe doing a lerp of projected gradients would be better
        #     return 0.5 * (p1 + p2)
        # else:

        coord = intersect_normals(
            extract_coord(self.vertices[i], axis),
            extract_coord(self.vertices[j], axis),
            extract_coord(self.vertex_gradients[i], axis),
            extract_coord(self.vertex_gradients[j], axis),
        )

        x_min, y_min, _ = (
            self.vertices[3] * (1 - SHRINK_FACTOR) + SHRINK_FACTOR * self.vertices[1]
        )
        x_max, y_max, _ = (
            self.vertices[3] * SHRINK_FACTOR + (1 - SHRINK_FACTOR) * self.vertices[1]
        )
        coord = np.clip(coord, [x_min, y_min][axis], [x_max, y_max][axis])
        # if self.vertices[0][0] == 1 and self.vertices[0][1] == 0:
        #     print(self.depth, "normals", normals)
        #     print("coord", coord)
        if axis == 0:
            return np.array([coord, self.vertices[i][1], 0])
        else:
            return np.array([self.vertices[i][0], coord, 0])
        # WHY AM I STRUGGLING? THIS WAS WORKING EARLIER
        # if self.vertices[0][0] == 1 and self.vertices[0][1] == 0 and self.depth == 2:
        #     print(i, j, derivative_1_2 / GRADIENT_EPS, derivative_2_1 / GRADIENT_EPS)
        # def intersect_normals(p, q, n, m):
        #     """Given vertices p,q and normals n,m, return the intersection coordinate between the lines passing
        #     through the vertices and perpendicular to the normals in XZ or YZ space. This coordinate would
        #     need to be combined with the other coordinate back in XY space."""
        #     return q[0] - m[1] * np.dot(n, q - p) / np.linalg.det([n, m])
        # n = np.array([derivative_1_2, -GRADIENT_EPS])
        # m = np.array([derivative_2_1, -GRADIENT_EPS])
        # p = np.array([p1[axis], p1[2]])
        # q = np.array([p2[axis], p2[2]])
        # int_coord = q[axis] - m[1] * np.dot(n, q - p) / np.linalg.det([n, m])
        # return np.array(
        #     [int_coord if axis == 0 else p1[0], int_coord if axis == 1 else p1[1], 0]
        # )

    def _get_edge_duals(self):
        return [
            self._apply_func_to(self._get_edge_dual_between(i, j))
            for i, j in CYCLIC_PAIRS
        ]

    def _get_face_dual(self):
        # assume self.vertex_gradients is already computed
        B = [np.dot(n, p) for n, p in zip(self.vertex_gradients, self.vertices)]

        x_min, y_min, _ = (
            self.vertices[3] * (1 - SHRINK_FACTOR) + SHRINK_FACTOR * self.vertices[1]
        )
        x_max, y_max, _ = (
            self.vertices[3] * SHRINK_FACTOR + (1 - SHRINK_FACTOR) * self.vertices[1]
        )
        result = scipy.optimize.lsq_linear(
            self.vertex_gradients,
            B,
            bounds=([x_min, y_min, -np.inf], [x_max, y_max, np.inf]),
        )
        return self._apply_func_to(result.x)

    def _compute_gradients(self):
        # gradients of G = f(x,y)-z
        # we compute these numerically based on a finite difference
        # TODO: compute these in a parent quad and pass it down
        # to reduce repeated computation
        self.vertex_gradients = [
            np.array(
                [
                    (self.fn(v[0] + GRADIENT_EPS_X, v[1]) - v[2]) / GRADIENT_EPS_X,
                    (self.fn(v[0], v[1] + GRADIENT_EPS_Y) - v[2]) / GRADIENT_EPS_Y,
                    -1,
                ]
            )
            for v in self.vertices
        ]

    def compute_duals(self):
        """Insert dual vertices. In the future, this should only be done for minimal cells"""
        self._compute_gradients()
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
